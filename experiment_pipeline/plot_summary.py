#!/usr/bin/env python
"""
Aggregate summary plots across all experiment configurations.

Reads metrics.json from each (n_peds, method) result directory and produces:
  - rmse_vs_npeds.png     : RMSE vs number of pedestrians (one line per method)
  - idsw_vs_npeds.png     : ID switches vs number of pedestrians (one line per method)

Usage:
  python plot_summary.py --results_dir experiment_pipeline/results
"""
from __future__ import print_function

import argparse
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiment_pipeline.config import METHODS, PED_COUNTS, N_TRIALS


def collect_metrics(results_dir):
    """Scan results directory and collect metrics for every (method, n_peds, trial).

    Returns:
      data[method_name][n_peds] = [metrics_dict, ...]  (one per trial)
    """
    data = {}
    for method in METHODS:
        mname = method["name"]
        data[mname] = {}
        for n in PED_COUNTS:
            data[mname][n] = []
            for trial in range(N_TRIALS):
                trial_dir = os.path.join(results_dir, f"trial_{trial}", f"n{n}", mname)
                metrics_path = os.path.join(trial_dir, "metrics.json")
                if os.path.isfile(metrics_path):
                    with open(metrics_path) as f:
                        data[mname][n].append(json.load(f))
    return data


def _mean_metric(metric_list, key):
    """Average a metric across trials, ignoring NaN."""
    vals = [m[key] for m in metric_list if key in m and np.isfinite(m[key])]
    return float(np.mean(vals)) if vals else float("nan")


def plot_rmse_vs_npeds(data, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v"]
    for i, method in enumerate(METHODS):
        mname = method["name"]
        label = method["label"]
        xs, ys = [], []
        for n in PED_COUNTS:
            if data[mname][n]:
                xs.append(n)
                ys.append(_mean_metric(data[mname][n], "total_rmse"))
        if xs:
            ax.plot(xs, ys, f"-{markers[i % len(markers)]}", label=label, linewidth=2, markersize=8)

    ax.set_xlabel("Number of Pedestrians", fontsize=12)
    ax.set_ylabel("RMSE (m)", fontsize=12)
    ax.set_title("RMSE vs Number of Pedestrians", fontsize=14)
    ax.set_xticks(PED_COUNTS)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved RMSE plot -> {save_path}")


def plot_idsw_vs_npeds(data, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v"]
    for i, method in enumerate(METHODS):
        mname = method["name"]
        label = method["label"]
        xs, ys = [], []
        for n in PED_COUNTS:
            if data[mname][n]:
                xs.append(n)
                ys.append(_mean_metric(data[mname][n], "id_switches"))
        if xs:
            ax.plot(xs, ys, f"-{markers[i % len(markers)]}", label=label, linewidth=2, markersize=8)

    ax.set_xlabel("Number of Pedestrians", fontsize=12)
    ax.set_ylabel("ID Switches", fontsize=12)
    ax.set_title("ID Switches vs Number of Pedestrians", fontsize=14)
    ax.set_xticks(PED_COUNTS)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved IDSW plot -> {save_path}")


def run_summary_plots(results_dir):
    """Generate all summary plots."""
    data = collect_metrics(results_dir)

    rmse_path = os.path.join(results_dir, "rmse_vs_npeds.png")
    idsw_path = os.path.join(results_dir, "idsw_vs_npeds.png")

    plot_rmse_vs_npeds(data, rmse_path)
    plot_idsw_vs_npeds(data, idsw_path)


def main():
    parser = argparse.ArgumentParser(description="Generate summary experiment plots")
    parser.add_argument("--results_dir", required=True, help="Root results directory")
    args = parser.parse_args()
    run_summary_plots(args.results_dir)


if __name__ == "__main__":
    main()
