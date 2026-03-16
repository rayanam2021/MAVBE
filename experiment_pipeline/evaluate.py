#!/usr/bin/env python
"""
Evaluation script for the experiment pipeline.

Loads ground-truth and predicted tracks in world frame, performs optimal
track-to-GT association via Hungarian algorithm on RMSE, computes metrics,
and generates trajectory + covariance-ellipse plots.

Outputs:
  - metrics.json       : RMSE, ID switches, per-track RMSE
  - trajectories.png   : bird's-eye XZ plot with GT, predictions, and covariance ellipses

Usage:
  python evaluate.py --gt gt_world.csv --pred tracks_world.csv --output_dir results/n3/imm_l05
"""
from __future__ import print_function

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiment_pipeline.config import COV_ELLIPSE_INTERVAL_FRAMES, FPS


# ── Data loading ──────────────────────────────────────────────────────

def load_gt(gt_path):
    """Load GT CSV: frame,ped_id,world_pX,world_pZ,world_pY,...
    Returns dict: {ped_id: [(frame, pX, pZ, pY), ...]} sorted by frame.
    Coordinates are in camera frame (pX=right, pZ=forward, pY=down).
    """
    trajectories = defaultdict(list)
    with open(gt_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["ped_id"])
            frame = int(row["frame"])
            pX = float(row["world_pX"])
            pZ = float(row["world_pZ"])
            pY = float(row["world_pY"])
            trajectories[pid].append((frame, pX, pZ, pY))
    for pid in trajectories:
        trajectories[pid].sort(key=lambda t: t[0])
    return dict(trajectories)


def load_predictions(pred_path):
    """Load predicted tracks CSV with world position and covariance.
    Returns dict: {track_id: [(frame, pX, pZ, pY, cov_3x3), ...]} sorted by frame.
    """
    trajectories = defaultdict(list)
    with open(pred_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = int(row["track_id"])
            frame = int(row["frame"])
            pX = float(row["world_pX"])
            pZ = float(row["world_pZ"])
            pY = float(row["world_pY"])
            cov = np.array([
                [float(row["cov_00"]), float(row["cov_01"]), float(row["cov_02"])],
                [float(row["cov_10"]), float(row["cov_11"]), float(row["cov_12"])],
                [float(row["cov_20"]), float(row["cov_21"]), float(row["cov_22"])],
            ])
            trajectories[tid].append((frame, pX, pZ, pY, cov))
    for tid in trajectories:
        trajectories[tid].sort(key=lambda t: t[0])
    return dict(trajectories)


# ── Track-to-GT matching ─────────────────────────────────────────────

def compute_pairwise_rmse(gt_trajs, pred_trajs):
    """Build cost matrix C[i,j] = RMSE between GT track i and pred track j
    over co-temporal frames.  Returns (gt_ids, pred_ids, cost_matrix).

    Both GT and predictions are in camera frame: (pX, pZ, pY).
    """
    gt_ids = sorted(gt_trajs.keys())
    pred_ids = sorted(pred_trajs.keys())
    n_gt = len(gt_ids)
    n_pred = len(pred_ids)

    cost = np.full((n_gt, n_pred), 1e6)

    for i, gid in enumerate(gt_ids):
        gt_by_frame = {f: (pX, pZ, pY) for f, pX, pZ, pY in gt_trajs[gid]}
        for j, pid in enumerate(pred_ids):
            dists_sq = []
            for f, pX, pZ, pY, _ in pred_trajs[pid]:
                if f in gt_by_frame:
                    gpX, gpZ, gpY = gt_by_frame[f]
                    dists_sq.append((gpX - pX) ** 2 + (gpZ - pZ) ** 2 + (gpY - pY) ** 2)
            if dists_sq:
                cost[i, j] = np.sqrt(np.mean(dists_sq))

    return gt_ids, pred_ids, cost


def hungarian_match(cost_matrix):
    """Optimal assignment using scipy Hungarian algorithm."""
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return list(zip(row_ind.tolist(), col_ind.tolist()))


# ── Metrics ───────────────────────────────────────────────────────────

def compute_metrics(gt_trajs, pred_trajs):
    """Compute RMSE and ID switches.

    Returns dict with:
      - total_rmse: overall RMSE across all matched pairs
      - per_track_rmse: {gt_id: rmse}
      - id_switches: count of extra predicted tracks beyond GT count +
                     per-frame ID reassignments
      - n_gt_tracks: number of GT pedestrians
      - n_pred_tracks: number of predicted tracks
      - match_map: {gt_id: pred_id}
    """
    if not gt_trajs or not pred_trajs:
        return {
            "total_rmse": float("nan"),
            "per_track_rmse": {},
            "id_switches": 0,
            "n_gt_tracks": len(gt_trajs),
            "n_pred_tracks": len(pred_trajs),
            "match_map": {},
        }

    gt_ids, pred_ids, cost = compute_pairwise_rmse(gt_trajs, pred_trajs)
    matches = hungarian_match(cost)

    match_map = {}
    per_track_rmse = {}
    all_dists_sq = []

    for gi, pi in matches:
        if cost[gi, pi] >= 1e5:
            continue
        gid = gt_ids[gi]
        pid = pred_ids[pi]
        match_map[gid] = pid
        per_track_rmse[gid] = cost[gi, pi]

        gt_by_frame = {f: (gpX, gpZ, gpY) for f, gpX, gpZ, gpY in gt_trajs[gid]}
        for f, pX, pZ, pY, _ in pred_trajs[pid]:
            if f in gt_by_frame:
                gpX, gpZ, gpY = gt_by_frame[f]
                all_dists_sq.append((gpX - pX) ** 2 + (gpZ - pZ) ** 2 + (gpY - pY) ** 2)

    total_rmse = np.sqrt(np.mean(all_dists_sq)) if all_dists_sq else float("nan")

    # ID switches: count extra tracks + per-frame reassignments
    n_extra = max(0, len(pred_ids) - len(gt_ids))

    # Per-frame ID consistency check: for each frame, find which pred track
    # is closest to each GT track and count deviations from global assignment
    frame_switches = 0
    matched_gt_ids = list(match_map.keys())
    if matched_gt_ids:
        gt_frame_data = {}
        for gid in matched_gt_ids:
            for f, gpX, gpZ, gpY in gt_trajs[gid]:
                gt_frame_data.setdefault(f, {})[gid] = (gpX, gpZ, gpY)

        pred_frame_data = {}
        for pid in pred_ids:
            for entry in pred_trajs[pid]:
                f = entry[0]
                pred_frame_data.setdefault(f, {})[pid] = (entry[1], entry[2], entry[3])

        last_assignment = {}
        for f in sorted(set(gt_frame_data.keys()) & set(pred_frame_data.keys())):
            for gid in matched_gt_ids:
                if gid not in gt_frame_data.get(f, {}):
                    continue
                gpX, gpZ, gpY = gt_frame_data[f][gid]
                best_pid = None
                best_dist = float("inf")
                for pid, (pX, pZ, pY) in pred_frame_data.get(f, {}).items():
                    d = (gpX - pX) ** 2 + (gpZ - pZ) ** 2 + (gpY - pY) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_pid = pid
                if best_pid is not None:
                    if gid in last_assignment and last_assignment[gid] != best_pid:
                        frame_switches += 1
                    last_assignment[gid] = best_pid

    id_switches = n_extra + frame_switches

    return {
        "total_rmse": float(total_rmse),
        "per_track_rmse": {int(k): float(v) for k, v in per_track_rmse.items()},
        "id_switches": int(id_switches),
        "n_gt_tracks": len(gt_ids),
        "n_pred_tracks": len(pred_ids),
        "match_map": {int(k): int(v) for k, v in match_map.items()},
    }


# ── Plotting ──────────────────────────────────────────────────────────

def _cov_ellipse(cov_2x2, n_std=2.0):
    """Return (width, height, angle_deg) for a 2D covariance ellipse."""
    vals, vecs = np.linalg.eigh(cov_2x2)
    vals = np.clip(vals, 0, None)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w = 2.0 * n_std * np.sqrt(vals[0])
    h = 2.0 * n_std * np.sqrt(vals[1])
    return w, h, angle


def plot_trajectories(gt_trajs, pred_trajs, match_map, pred_full, save_path,
                      method_label=""):
    """Plot GT and predicted trajectories in the XZ (ground) plane with covariance ellipses.

    gt_trajs:   {ped_id: [(frame, x, y, z), ...]}
    pred_trajs: {track_id: [(frame, pX, pZ, pY, cov_3x3), ...]}
    match_map:  {gt_id: pred_id}
    pred_full:  same as pred_trajs (full data)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.tab10

    # Plot GT trajectories (pX = right, pZ = forward in camera frame)
    for idx, (gid, points) in enumerate(sorted(gt_trajs.items())):
        color = cmap(idx % 10)
        xs = [p[1] for p in points]  # pX (camera-right)
        ys = [p[2] for p in points]  # pZ (camera-forward)
        ax.plot(xs, ys, "-", color=color, linewidth=2, label=f"GT {gid}")

    # Plot predicted trajectories (matched)
    for idx, (gid, pid) in enumerate(sorted(match_map.items())):
        color = cmap(idx % 10)
        if pid not in pred_trajs:
            continue
        points = pred_trajs[pid]
        xs = [p[1] for p in points]  # pX
        ys = [p[2] for p in points]  # pZ
        ax.plot(xs, ys, "--", color=color, linewidth=1.5, alpha=0.8,
                label=f"Pred {pid} (→GT {gid})")

        # Covariance ellipses every COV_ELLIPSE_INTERVAL_FRAMES
        for entry in points:
            f, pX, pZ, pY, cov = entry
            if f % COV_ELLIPSE_INTERVAL_FRAMES != 0:
                continue
            # XZ submatrix: rows/cols 0 (pX) and 1 (pZ) of the 3x3
            cov_xz = cov[np.ix_([0, 1], [0, 1])]
            try:
                w, h, angle = _cov_ellipse(cov_xz, n_std=2.0)
                if np.isfinite(w) and np.isfinite(h) and w < 100 and h < 100:
                    e = Ellipse((pX, pZ), w, h, angle=angle,
                                facecolor=color, alpha=0.15, edgecolor=color, linewidth=0.5)
                    ax.add_patch(e)
            except Exception:
                pass

    # Plot unmatched predicted tracks
    matched_pids = set(match_map.values())
    for pid, points in sorted(pred_full.items()):
        if pid in matched_pids:
            continue
        xs = [p[1] for p in points]
        ys = [p[2] for p in points]
        ax.plot(xs, ys, ":", color="gray", linewidth=1, alpha=0.5,
                label=f"Pred {pid} (unmatched)")

    ax.set_xlabel("pX — camera right (m)")
    ax.set_ylabel("pZ — camera forward (m)")
    title = "Trajectories: GT vs Predicted"
    if method_label:
        title += f"  [{method_label}]"
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Saved trajectory plot -> {save_path}")


# ── Main evaluation function ─────────────────────────────────────────

def run_evaluation(gt_path, pred_path, output_dir, method_label=""):
    """Run full evaluation: matching, metrics, plots."""
    os.makedirs(output_dir, exist_ok=True)

    gt_trajs = load_gt(gt_path)
    pred_trajs = load_predictions(pred_path)

    print(f"[evaluate] GT tracks: {len(gt_trajs)}  Pred tracks: {len(pred_trajs)}")

    metrics = compute_metrics(gt_trajs, pred_trajs)

    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] RMSE={metrics['total_rmse']:.4f}  "
          f"IDSW={metrics['id_switches']}  "
          f"GT={metrics['n_gt_tracks']}  Pred={metrics['n_pred_tracks']}")

    # Plot
    traj_path = os.path.join(output_dir, "trajectories.png")
    plot_trajectories(
        gt_trajs, pred_trajs,
        metrics["match_map"], pred_trajs,
        traj_path, method_label=method_label,
    )

    return metrics


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate tracking vs GT in world frame")
    parser.add_argument("--gt", required=True, help="Path to gt_world.csv")
    parser.add_argument("--pred", required=True, help="Path to tracks_world.csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label", default="", help="Method label for plot title")
    args = parser.parse_args()

    run_evaluation(args.gt, args.pred, args.output_dir, method_label=args.label)


if __name__ == "__main__":
    main()
