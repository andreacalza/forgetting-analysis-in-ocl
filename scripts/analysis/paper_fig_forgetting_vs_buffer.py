#!/usr/bin/env python3
"""epochs_forgetting.png for the paper, with only these changes:
  1) shallow panel uses the paper definition F_shallow = F_out - F_deep
     (computed per seed, then mean +/- std over seeds; analyze_epochs plots F_out there)
  2) "epoch(s)" -> "update(s)", y labels as in the tables, larger text
Layout, colours, markers, axes and data are those of analyze_epochs.plot_metric_panels.

Usage:
    python scripts/analysis/paper_fig_forgetting_vs_buffer.py \
        results_ep1 results_ep3 results_ep5 results_ep50 --out paper_figs/forgetting_vs_buffer
"""
import argparse
import csv
import os
import sys
from statistics import mean, stdev

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyze_epochs import EPOCH_STYLE, GRAY, buffer_axis, reference_final_step, scan  # noqa: E402


def ms(v):
    return mean(v), (stdev(v) if len(v) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--probe-file", default="lp_curve_sgd.csv",
                    choices=["lp_curve_sgd.csv", "lp_curve.csv"])
    ap.add_argument("--out", default="paper_figs/forgetting_vs_buffer",
                    help="output path without extension (.png, .pdf and .csv are written)")
    ap.add_argument("--fontsize", type=float, default=16)
    args = ap.parse_args()

    fs = args.fontsize
    plt.rcParams.update({"font.size": fs, "axes.titlesize": fs + 1, "axes.labelsize": fs,
                         "xtick.labelsize": fs - 3, "ytick.labelsize": fs - 2,
                         "legend.fontsize": fs - 4, "figure.titlesize": fs + 2})

    records = scan(args.roots)
    ref = reference_final_step(records)

    data = {}   # (u, buf) -> per-seed lists
    for r in records:
        cs = r["curves"].get(args.probe_file)
        target = ref.get(r["ep"])
        if not cs or (target and cs[3] < target):
            print(f"[skip] {r['dir']}: no complete {args.probe_file}")
            continue
        deep = cs[2] * 100           # F_deep: mean over tasks of a*_jj - a*_Tj
        out = r["shallow"]           # F_out : mean over tasks of a_jj - a_Tj  (pp)
        e = data.setdefault((r["ep"], r["buf"]), {"out": [], "deep": [], "shallow": []})
        e["out"].append(out); e["deep"].append(deep); e["shallow"].append(out - deep)

    budgets = sorted({u for u, _ in data})
    buffers = sorted({b for _, b in data})
    ers = [b for b in buffers if b > 0]
    naive_x = ers[0] * 0.45

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(f"{args.out}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["u", "buffer", "n_seeds", "F_out", "F_deep", "F_shallow"])
        for (u, b), e in sorted(data.items()):
            w.writerow([u, b, len(e["out"])] + ["{:+.1f}±{:.1f}".format(*ms(e[k]))
                                                for k in ("out", "deep", "shallow")])
    print(f"-> {args.out}.csv")

    probe = "SGD" if args.probe_file == "lp_curve_sgd.csv" else "logreg"
    panels = [("shallow", r"$F^{\mathrm{shallow}}$ (pp)", "Shallow (output layer)"),
              ("deep", r"$F^{\mathrm{deep}}$ (pp)", f"Deep (linear probe, {probe})")]
    fig, axes = plt.subplots(1, 2, figsize=(6.2 * 2, 5.2))
    for ax, (key, ylabel, title) in zip(axes, panels):
        for u in budgets:
            col, mk = EPOCH_STYLE.get(u, (GRAY, "o"))
            xs, m, s = [], [], []
            for b in buffers:
                e = data.get((u, b))
                if e:
                    mm, ss = ms(e[key])
                    xs.append(naive_x if b == 0 else b); m.append(mm); s.append(ss)
            ax.errorbar(xs, m, yerr=s, marker=mk, ms=6, lw=1.8, capsize=3, color=col,
                        label=f"{u} update" + ("s" if u > 1 else ""))
        ax.axhline(0, lw=.8, color="#999999")
        buffer_axis(ax, buffers, naive_x)
        for lab in ax.get_xticklabels():
            lab.set_fontsize(fs - 4)
        ax.set(xlabel="buffer size (0 = Naive)", ylabel=ylabel, title=title)
        ax.legend(frameon=False)
    fig.suptitle("Forgetting vs buffer size, across update budgets")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight")
        print(f"-> {args.out}.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
