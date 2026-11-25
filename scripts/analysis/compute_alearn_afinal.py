#!/usr/bin/env python3
"""a_learn and a_final of the paper, from the linear-probe curves (lp_curve_sgd.csv):

    a_learn = mean over j = 0..T-2 of a*_jj   (probe accuracy on task j right after learning it)
    a_final = mean over j = 0..T-2 of a*_Tj   (probe accuracy on task j at the end of the stream)
    so that F_deep = a_learn - a_final        (same tasks, same points as analyze_epochs)

'right after learning it' = last checkpoint at which j is the newest task, as in
compute_lp_sgd.py / analyze_epochs.py. Per run, then mean +/- std over seeds.

Usage:
    python scripts/analysis/compute_alearn_afinal.py results_ep1 results_ep3 results_ep5 results_ep50
Output: analysis_out_alearn/alearn_afinal.csv (and printed)
"""
import argparse
import csv
import glob
import os
import re
from statistics import mean, stdev


def ms(v):
    return mean(v), (stdev(v) if len(v) > 1 else 0.0)


def buffer_of(folder):
    if folder.lower().startswith("naive"):
        return 0
    return int(re.search(r"(\d+)$", folder).group(1))


def epochs_of(run_dir):
    m = re.search(r"train_epochs:\s*(\d+)", open(os.path.join(run_dir, "config.yaml")).read())
    return int(m.group(1))


def learn_final(path):
    acc = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            acc[(int(r["step"]), int(r["task"]))] = float(r["probe_acc"]) * 100
    steps = sorted({s for s, _ in acc})
    last = steps[-1]
    newest_at = {s: max(t for (ss, t) in acc if ss == s) for s in steps}
    tasks = sorted({t for _, t in acc})
    learn, final = [], []
    for t in tasks[:-1]:
        own = [s for s in steps if newest_at[s] == t and (s, t) in acc]
        if own and (last, t) in acc:
            learn.append(acc[(max(own), t)])
            final.append(acc[(last, t)])
    return mean(learn), mean(final), last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--probe-file", default="lp_curve_sgd.csv")
    ap.add_argument("--outdir", default="analysis_out_alearn")
    args = ap.parse_args()

    cells, last_steps = {}, {}
    for root in args.roots:
        for run in sorted(glob.glob(os.path.join(root, "*", "[0-9]"))):
            p = os.path.join(run, args.probe_file)
            if not os.path.isfile(p):
                print(f"[skip] {run}: no {args.probe_file}")
                continue
            u, b = epochs_of(run), buffer_of(os.path.basename(os.path.dirname(run)))
            a_l, a_f, last = learn_final(p)
            cells.setdefault((u, b), []).append((a_l, a_f, run))
            last_steps.setdefault(u, set()).add(last)
    for u, s in last_steps.items():
        if len(s) > 1:
            print(f"WARNING u={u}: curves end at different steps {sorted(s)} (truncated runs?)")

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "alearn_afinal.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["u", "buffer", "n_seeds", "a_learn", "a_final", "F_deep"])
        print(f"{'u':>3} {'buffer':>7} {'n':>2}  {'a_learn':>12} {'a_final':>12} {'F_deep':>12}")
        for (u, b), v in sorted(cells.items()):
            al, af = [x[0] for x in v], [x[1] for x in v]
            fd = [x - y for x, y in zip(al, af)]
            row = ["{:.1f}±{:.1f}".format(*ms(al)), "{:.1f}±{:.1f}".format(*ms(af)),
                   "{:+.1f}±{:.1f}".format(*ms(fd))]
            w.writerow([u, b, len(v)] + row)
            print(f"{u:>3} {b:>7} {len(v):>2}  {row[0]:>12} {row[1]:>12} {row[2]:>12}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
