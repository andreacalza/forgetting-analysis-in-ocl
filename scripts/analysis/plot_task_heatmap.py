#!/usr/bin/env python3
"""Per-task accuracy matrix heatmaps.

Builds R[j][i] = the metric on task i measured right after finishing task j
(lower-triangular matrix: a task is only evaluated once it has been
trained).

  * Output (Top-1) accuracy -- from logs.json                (always available)
  * Linear-probe accuracy   -- from lp_curve.csv              (if present)
  * Linear-probe (SGD) acc. -- from lp_curve_sgd.csv           (if present)
  * SNR                      -- from snr_curve.csv             (if present)

"""
import argparse
import csv
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from compute_forgetting import EXP_KEY  


def shallow_matrix(run_dir):
    """R[j][i] = Top-1 accuracy on task i right after task j."""
    logs_path = os.path.join(run_dir, "logs.json")
    if not os.path.exists(logs_path):
        return None
    last_at_level = {}
    with open(logs_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            accs = {}
            for k, v in d.items():
                m = EXP_KEY.match(k)
                if m:
                    accs[int(m.group(1))] = v
            if accs:
                last_at_level[max(accs)] = accs
    if not last_at_level:
        return None
    n = max(last_at_level) + 1
    R = np.full((n, n), np.nan)
    for j, accs in last_at_level.items():
        for i, a in accs.items():
            R[j, i] = a
    return R


def csv_matrix(csv_path):
    """R[j][i] = value on task i at the last checkpoint where j is the newest
    task trained.
    """
    if not os.path.exists(csv_path):
        return None
    rows = []
    with open(csv_path) as f:
        r = csv.reader(f)
        header = next(r)
        val_col = header[-1]  
        for row in r:
            s, t = int(row[0]), int(row[1])
            v = float(row[-1])
            rows.append((s, t, v))
    if not rows:
        return None
    acc = {}
    by_step = {}
    for s, t, v in rows:
        acc[(s, t)] = v
        by_step.setdefault(s, set()).add(t)
    n = max(t for _, t, _ in rows) + 1
    R = np.full((n, n), np.nan)
    for j in range(n):
        # last step at which j is the newest (highest) task recorded
        cand = [s for s, tasks in by_step.items() if max(tasks) == j]
        if not cand:
            continue
        step = max(cand)
        for i in by_step[step]:
            R[j, i] = acc[(step, i)]
    return R, val_col


def plot_heatmaps(matrices, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrices = [(name, R, cmap, vmin, vmax) for name, R, cmap, vmin, vmax in matrices if R is not None]
    if not matrices:
        print("No data found for any metric - nothing to plot.")
        return

    fig, axes = plt.subplots(1, len(matrices), figsize=(5.2 * len(matrices), 4.6))
    if len(matrices) == 1:
        axes = [axes]

    for ax, (name, R, cmap, vmin, vmax) in zip(axes, matrices):
        masked = np.ma.masked_invalid(R)
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xlabel("evaluated task $i$")
        ax.set_ylabel("trained up to task $j$")
        ax.set_title(name)
        n = R.shape[0]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Per-task accuracy matrix $R[j][i]$ (lower-triangular: task $i$ evaluated only once trained)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"-> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lp-file", default="lp_curve.csv")
    ap.add_argument("--lp-sgd-file", default="lp_curve_sgd.csv")
    ap.add_argument("--snr-file", default="snr_curve.csv")
    ap.add_argument("--absolute-scale", action="store_true",
                    help="colour-scale the accuracy panels over the full 0-1 range "
                         "instead of the data range. Comparable across figures, but "
                         "it flattens the within-figure contrast: if every accuracy "
                         "sits between .5 and .7 the whole matrix looks uniform and "
                         "the forgetting pattern -- the point of the plot -- vanishes.")
    args = ap.parse_args()

    acc_lo, acc_hi = (0.0, 1.0) if args.absolute_scale else (None, None)

    matrices = []

    R_shallow = shallow_matrix(args.run_dir)
    matrices.append(("Output (Top-1) accuracy", R_shallow, "viridis", acc_lo, acc_hi))

    lp = csv_matrix(os.path.join(args.run_dir, args.lp_file))
    if lp is not None:
        matrices.append(("Linear-probe accuracy", lp[0], "viridis", acc_lo, acc_hi))

    lp_sgd = csv_matrix(os.path.join(args.run_dir, args.lp_sgd_file))
    if lp_sgd is not None:
        matrices.append(("Linear-probe accuracy (SGD)", lp_sgd[0], "viridis", acc_lo, acc_hi))

    snr = csv_matrix(os.path.join(args.run_dir, args.snr_file))
    if snr is not None:
        matrices.append(("SNR", snr[0], "magma", None, None))

    out = args.out or os.path.join(args.run_dir, "task_heatmap.png")
    plot_heatmaps(matrices, out)


if __name__ == "__main__":
    main()
