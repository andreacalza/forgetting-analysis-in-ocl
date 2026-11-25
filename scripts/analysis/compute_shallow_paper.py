#!/usr/bin/env python3
"""Shallow forgetting as defined in the paper, Eq. (1):

    F_out_j     = a_jj  - a_Tj     model's own (Top-1) accuracy, from logs.json
    F_deep_j    = a*_jj - a*_Tj    linear probe, from lp_curve_sgd.csv (or lp_curve.csv)
    F_shallow_j = F_out_j - F_deep_j

Same data and same 'learned'/'final' points as analyze_epochs.py (its shallow_stats and
curve_stats). Per task j = 0..T-2, then mean over tasks per run, then mean +/- std over
seeds. Nothing is recomputed from checkpoints.

Outputs (in --outdir):
  shallow_paper_summary.csv   u, buffer, n_seeds, F_out, F_deep, F_shallow (mean±std over seeds)
  shallow_paper_per_run.csv   one row per run (seed): the three task-averaged values
  shallow_paper_per_task.csv  one row per (run, task): the three per-task values

Usage:
    python scripts/analysis/compute_shallow_paper.py results_ep1 results_ep3 results_ep5 results_ep50 \
        --outdir analysis_out_shallow
"""
import argparse
import csv
import json
import os
import sys
from statistics import mean, stdev

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_epochs import EXP, reference_final_step, scan  # noqa: E402


def output_matrix(logs_json):
    """R[j][i] = Top-1 accuracy on task i at the last evaluation while task j is the newest."""
    last = {}
    with open(logs_json) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            accs = {int(m.group(1)): v for k, v in d.items() for m in [EXP.match(k)] if m}
            if accs:
                last[max(accs)] = accs
    return last


def ms(v):
    return mean(v), (stdev(v) if len(v) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--probe-file", default="lp_curve_sgd.csv",
                    choices=["lp_curve_sgd.csv", "lp_curve.csv"])
    ap.add_argument("--outdir", default="analysis_out_shallow")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    records = scan(args.roots)
    ref = reference_final_step(records)

    per_task, per_run = [], []
    for r in records:
        cs = r["curves"].get(args.probe_file)
        target = ref.get(r["ep"])
        if not cs or (target and cs[3] < target):
            print(f"[skip] {r['dir']}: no complete {args.probe_file}")
            continue
        R = output_matrix(os.path.join(r["dir"], "logs.json"))
        T = max(R) + 1
        deep = cs[4]                                   # {task: a*_jj - a*_Tj}
        rows = []
        for j in range(T - 1):
            if j not in deep or j not in R.get(j, {}) or j not in R.get(T - 1, {}):
                continue
            f_out = (R[j][j] - R[T - 1][j]) * 100
            f_deep = deep[j] * 100
            rows.append((j, f_out, f_deep, f_out - f_deep))
        if not rows:
            continue
        seed = os.path.basename(r["dir"])
        for j, fo, fd, fs in rows:
            per_task.append([r["ep"], r["buf"], seed, j, fo, fd, fs])
        per_run.append([r["ep"], r["buf"], seed,
                        mean(x[1] for x in rows), mean(x[2] for x in rows),
                        mean(x[3] for x in rows)])

    with open(os.path.join(args.outdir, "shallow_paper_per_task.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["u", "buffer", "seed", "task", "F_out", "F_deep", "F_shallow"])
        for row in sorted(per_task, key=lambda x: (x[0], x[1], x[2], x[3])):
            w.writerow(row[:4] + [f"{v:.3f}" for v in row[4:]])

    with open(os.path.join(args.outdir, "shallow_paper_per_run.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["u", "buffer", "seed", "F_out", "F_deep", "F_shallow"])
        for row in sorted(per_run, key=lambda x: (x[0], x[1], x[2])):
            w.writerow(row[:3] + [f"{v:.3f}" for v in row[3:]])

    cells = {}
    for u, b, _, fo, fd, fs in per_run:
        c = cells.setdefault((u, b), {"out": [], "deep": [], "shallow": []})
        c["out"].append(fo); c["deep"].append(fd); c["shallow"].append(fs)
    with open(os.path.join(args.outdir, "shallow_paper_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["u", "buffer", "n_seeds", "F_out", "F_deep", "F_shallow"])
        for (u, b), c in sorted(cells.items()):
            w.writerow([u, b, len(c["out"])] + ["{:+.1f}±{:.1f}".format(*ms(c[k]))
                                                for k in ("out", "deep", "shallow")])

    print(f"{len(per_run)} runs")
    for n in ("shallow_paper_summary.csv", "shallow_paper_per_run.csv", "shallow_paper_per_task.csv"):
        print(f"-> {os.path.join(args.outdir, n)}")


if __name__ == "__main__":
    main()
