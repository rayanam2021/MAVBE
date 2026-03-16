#!/usr/bin/env python
"""
Compute MOT and trajectory metrics from MOTChallenge-format files or from Python.

Usage (MOT files):
  python evaluation/run_metrics.py --gt path/to/gt/gt.txt --pred path/to/results.txt [--out report.txt]
  python evaluation/run_metrics.py --gt gt.txt --pred pred.txt --plot trajectories.png

Usage (from Python):
  from evaluation.metrics import compute_metrics
  from evaluation.io import load_mot_file
  gt_frames = load_mot_file("gt/gt.txt")
  pred_frames = load_mot_file("results.txt")
  m = compute_metrics(gt_frames, pred_frames)
"""
from __future__ import print_function

import argparse
import sys
from pathlib import Path

# Allow running from repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import compute_metrics, format_metrics
from evaluation.io import load_mot_file, save_metrics_report


def _frames_to_trajectories(frames):
    """Build id -> [(cx, cy), ...] in frame order from frames (list of list of (id, bbox_xyxy))."""
    from collections import defaultdict
    trajectories = defaultdict(list)
    for frame_list in frames:
        for tid, bbox in frame_list:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            trajectories[tid].append((cx, cy))
    return dict(trajectories)


def plot_trajectories(gt_frames, pred_frames, save_path):
    """Plot predicted and ground-truth trajectories (2D centers) to save_path."""
    import numpy as np
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping trajectory plot")
        return
    gt_traj = _frames_to_trajectories(gt_frames)
    pred_traj = _frames_to_trajectories(pred_frames)
    plt.figure(figsize=(12, 8))
    for tid, points in pred_traj.items():
        if not points:
            continue
        points = np.array(points)
        plt.plot(points[:, 0], points[:, 1], marker="o", markersize=3, label="Pred ID %d" % tid)
    for tid, points in gt_traj.items():
        if not points:
            continue
        points = np.array(points)
        plt.plot(
            points[:, 0], points[:, 1],
            linestyle="--", marker="s", markersize=2, alpha=0.8,
            label="GT ID %d" % tid,
        )
    plt.gca().invert_yaxis()
    plt.xlabel("X pixels")
    plt.ylabel("Y pixels")
    plt.title("Trajectories (predicted and ground truth)")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print("Trajectory plot saved to %s" % save_path)


def main():
    parser = argparse.ArgumentParser(description="Compute MOT and trajectory metrics")
    parser.add_argument("--gt", required=True, help="Path to ground truth file (MOT format: frame,id,left,top,width,height)")
    parser.add_argument("--pred", required=True, help="Path to tracker output (same format)")
    parser.add_argument("--out", default="", help="Optional: write report to this file")
    parser.add_argument("--plot", default="", help="Optional: path to save trajectory comparison PNG (pred + GT)")
    parser.add_argument("--min-iou", type=float, default=0.5, help="Min IoU for match (default 0.5)")
    args = parser.parse_args()

    gt_path = Path(args.gt)
    pred_path = Path(args.pred)
    if not gt_path.exists():
        print("GT file not found: %s" % gt_path)
        sys.exit(1)
    if not pred_path.exists():
        print("Pred file not found: %s" % pred_path)
        sys.exit(1)

    print("Loading GT: %s" % gt_path)
    gt_frames = load_mot_file(str(gt_path))
    print("Loading pred: %s" % pred_path)
    pred_frames = load_mot_file(str(pred_path))
    print("GT frames: %d, pred frames: %d" % (len(gt_frames), len(pred_frames)))

    metrics = compute_metrics(gt_frames, pred_frames, min_iou=args.min_iou)
    print("")
    print(format_metrics(metrics))

    if args.out:
        save_metrics_report(metrics, args.out)

    if args.plot:
        plot_trajectories(gt_frames, pred_frames, args.plot)


if __name__ == "__main__":
    main()
