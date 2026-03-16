#!/usr/bin/env python
"""
Main experiment orchestrator.

Pipeline order (phases are fully decoupled):
  Phase 1 — CARLA recording: generate all videos, depth frames, and GT.
            CARLA can be shut down after this phase completes.
  Phase 2 — Tracking: run YOLO + DeepSORT with each filter/lambda config.
  Phase 3 — Evaluation: compute RMSE, IDSW, trajectory plots per method.
  Phase 4 — Summary plots: aggregate results across all scenarios.

Usage:
  python run_experiments.py                          # default: all phases, single trial
  python run_experiments.py --trials 3               # 3 trials per config
  python run_experiments.py --skip_carla             # reuse existing videos/GT
  python run_experiments.py --skip_tracking          # reuse existing tracking outputs
  python run_experiments.py --only_plots             # only regenerate summary plots
  python run_experiments.py --results_dir my_results # custom output directory
"""
from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiment_pipeline.config import (
    METHODS, PED_COUNTS, N_TRIALS,
    FX, FY, CX, CY, CARLA_HOST, CARLA_PORT,
)
from experiment_pipeline.carla_scenario import run_scenario
from experiment_pipeline.run_tracking import run_tracking
from experiment_pipeline.evaluate import run_evaluation
from experiment_pipeline.plot_summary import run_summary_plots


def _scenario_dirs(results_dir, n_trials):
    """Yield (trial, n_peds, trial_dir, seed) for every scenario."""
    for trial in range(n_trials):
        for n_peds in PED_COUNTS:
            trial_dir = os.path.join(results_dir, f"trial_{trial}", f"n{n_peds}")
            seed = trial * 1000 + n_peds
            yield trial, n_peds, trial_dir, seed


def run_all(results_dir, n_trials=N_TRIALS,
            skip_carla=False, skip_tracking=False,
            only_plots=False, only_carla=False,
            device_str="", weights=None, save_video=False,
            host=CARLA_HOST, port=CARLA_PORT):
    """Execute the full experiment sweep."""

    os.makedirs(results_dir, exist_ok=True)
    t0 = time.time()

    if only_plots:
        print("=== Only regenerating summary plots ===")
        run_summary_plots(results_dir)
        return

    scenarios = list(_scenario_dirs(results_dir, n_trials))
    total_scenarios = len(scenarios)
    total_tracking = total_scenarios * len(METHODS)

    # ──────────────────────────────────────────────────────────────
    #  Phase 1: CARLA recording  (all videos / depth / GT first)
    # ──────────────────────────────────────────────────────────────
    recorded_dirs = []  # trial_dirs that have valid recordings

    if not skip_carla:
        print(f"\n{'='*60}")
        print(f"  PHASE 1 — CARLA RECORDING  ({total_scenarios} scenarios)")
        print(f"{'='*60}")
        for idx, (trial, n_peds, trial_dir, seed) in enumerate(scenarios, 1):
            print(f"\n[CARLA {idx}/{total_scenarios}] trial={trial}  n_peds={n_peds}  seed={seed}")
            try:
                run_scenario(
                    n_peds=n_peds,
                    seed=seed,
                    output_dir=trial_dir,
                    host=host,
                    port=port,
                )
                recorded_dirs.append(trial_dir)
            except Exception as e:
                print(f"[ERROR] CARLA scenario failed: {e}")

        carla_elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"  PHASE 1 COMPLETE — {len(recorded_dirs)}/{total_scenarios} "
              f"scenarios recorded in {carla_elapsed:.1f}s")
        print(f"  CARLA is no longer needed. You may shut it down.")
        print(f"{'='*60}")
    else:
        for _, _, trial_dir, _ in scenarios:
            video_path = os.path.join(trial_dir, "video.mp4")
            if os.path.isfile(video_path):
                recorded_dirs.append(trial_dir)
            else:
                print(f"[SKIP] No video at {video_path}")
        print(f"[CARLA] Skipped — found {len(recorded_dirs)} existing recordings")

    if only_carla:
        print(f"\n--only_carla set. Stopping after Phase 1.")
        print(f"Results directory: {os.path.abspath(results_dir)}")
        return

    # ──────────────────────────────────────────────────────────────
    #  Phase 2: Tracking  (all methods on all recorded scenarios)
    # ──────────────────────────────────────────────────────────────
    done = 0
    t1 = time.time()

    if not skip_tracking:
        print(f"\n{'='*60}")
        print(f"  PHASE 2 — TRACKING  ({len(recorded_dirs)} scenarios x "
              f"{len(METHODS)} methods = {len(recorded_dirs) * len(METHODS)} runs)")
        print(f"{'='*60}")

    for trial_dir in recorded_dirs:
        video_path = os.path.join(trial_dir, "video.mp4")
        depth_dir = os.path.join(trial_dir, "depth_frames")

        for method in METHODS:
            mname = method["name"]
            ftype = method["filter"]
            lam = method["lambda_"]
            method_dir = os.path.join(trial_dir, mname)

            done += 1
            rel = os.path.relpath(trial_dir, results_dir)
            print(f"\n--- [Track {done}/{total_tracking}] {rel}/{mname} ---")

            if not skip_tracking:
                try:
                    run_tracking(
                        source=video_path,
                        depth_dir=depth_dir,
                        filter_type=ftype,
                        lambda_val=lam,
                        output_dir=method_dir,
                        weights=weights,
                        device_str=device_str,
                        fx=FX, fy=FY, cx=CX, cy=CY,
                        save_video=save_video,
                    )
                except Exception as e:
                    print(f"[ERROR] Tracking failed: {e}")
            else:
                tracks_path = os.path.join(method_dir, "tracks_world.csv")
                if not os.path.isfile(tracks_path):
                    print(f"[SKIP] No tracks at {tracks_path}")

    if not skip_tracking:
        track_elapsed = time.time() - t1
        print(f"\n  PHASE 2 COMPLETE — tracking finished in {track_elapsed:.1f}s")

    # ──────────────────────────────────────────────────────────────
    #  Phase 3: Evaluation  (metrics + plots per method)
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 3 — EVALUATION")
    print(f"{'='*60}")

    for trial_dir in recorded_dirs:
        gt_path = os.path.join(trial_dir, "gt_world.csv")

        for method in METHODS:
            mname = method["name"]
            label = method["label"]
            method_dir = os.path.join(trial_dir, mname)
            pred_path = os.path.join(method_dir, "tracks_world.csv")

            if os.path.isfile(pred_path) and os.path.isfile(gt_path):
                try:
                    run_evaluation(
                        gt_path=gt_path,
                        pred_path=pred_path,
                        output_dir=method_dir,
                        method_label=label,
                    )
                except Exception as e:
                    print(f"[ERROR] Evaluation failed for {method_dir}: {e}")
            else:
                rel = os.path.relpath(method_dir, results_dir)
                print(f"[SKIP] Missing files for {rel}")

    # ──────────────────────────────────────────────────────────────
    #  Phase 4: Summary plots
    # ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  PHASE 4 — SUMMARY PLOTS  (total elapsed: {elapsed:.1f}s)")
    print(f"{'='*60}")
    run_summary_plots(results_dir)
    print(f"\nResults directory: {os.path.abspath(results_dir)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run full experiment pipeline: CARLA -> tracking -> evaluation -> plots"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--results_dir",
                        default=os.path.join(_SCRIPT_DIR, "results", f"run_{timestamp}"),
                        help="Root output directory (auto-timestamped by default)")
    parser.add_argument("--trials", type=int, default=N_TRIALS,
                        help="Number of random-seed trials per config")
    parser.add_argument("--skip_carla", action="store_true",
                        help="Skip CARLA recording (reuse existing videos)")
    parser.add_argument("--skip_tracking", action="store_true",
                        help="Skip tracking (reuse existing track files)")
    parser.add_argument("--only_carla", action="store_true",
                        help="Only run CARLA recording (Phase 1), then stop")
    parser.add_argument("--only_plots", action="store_true",
                        help="Only regenerate summary plots from existing metrics")
    parser.add_argument("--device", default="",
                        help="Torch device for tracking (e.g. cuda:0)")
    parser.add_argument("--weights", default=None,
                        help="Path to YOLO weights file")
    parser.add_argument("--save_video", action="store_true",
                        help="Save tracking video with bounding boxes per method")
    parser.add_argument("--host", default=CARLA_HOST)
    parser.add_argument("--port", type=int, default=CARLA_PORT)
    args = parser.parse_args()

    run_all(
        results_dir=args.results_dir,
        n_trials=args.trials,
        skip_carla=args.skip_carla,
        skip_tracking=args.skip_tracking or args.only_carla,
        only_plots=args.only_plots,
        only_carla=args.only_carla,
        device_str=args.device,
        weights=args.weights,
        save_video=args.save_video,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
