#!/usr/bin/env python3
"""Paper Fig. 2: deep forgetting by task position, one panel per update budget u, one curve
per buffer size. Data and computation are those of analyze_epochs.plot_per_task_by_buffer.

Paper version (u = 1 and u = 50 side by side, one legend per panel, bottom right):
    python scripts/analysis/paper_fig_deep_by_task.py results_ep1 results_ep50 --no-suptitle \
        --fontsize 16 --xlabel "task index (order in the stream)" \
        --legend-in each --legend-loc "lower right" --legend-ncols 2 \
        --panel-width 5.8 --panel-height 4.6 --out paper_figs/fig2_deep_by_task
"""
import argparse
import os
import sys
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyze_epochs import GRAY, reference_final_step, scan  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--probe-file", default="lp_curve_sgd.csv",
                    choices=["lp_curve_sgd.csv", "lp_curve.csv"])
    ap.add_argument("--out", default="paper_figs/deep_by_task",
                    help="output path without extension (.png and .pdf are written)")
    ap.add_argument("--fontsize", type=float, default=20)
    ap.add_argument("--color-buffers", default="200,500,1000,2000,5000,10000,50000",
                    help="fixed buffer list for the colour scale, so the same buffer has the "
                         "same colour in every figure (legend still lists only plotted ones)")
    ap.add_argument("--no-suptitle", action="store_true")
    ap.add_argument("--ylim", default=None,
                    help="fixed y range 'min,max' (same in every figure, so they are comparable)")
    ap.add_argument("--xlabel", default="task index")
    ap.add_argument("--legend-in", choices=["first", "last", "each"], default="last",
                    help="panel that holds the legend; each = one legend per panel "
                         "listing only the buffers of that panel")
    ap.add_argument("--legend-loc", default="best")
    ap.add_argument("--legend-ncols", default="2",
                    help="columns of the legend; with --legend-in each, one value per panel, e.g. 4,2")
    ap.add_argument("--legend-fontsize", type=float, default=None)
    ap.add_argument("--panel-width", type=float, default=5.8, help="inches per panel")
    ap.add_argument("--panel-height", type=float, default=5.8, help="inches")
    args = ap.parse_args()

    fs = args.fontsize
    plt.rcParams.update({"font.size": fs, "axes.titlesize": fs + 1, "axes.labelsize": fs,
                         "xtick.labelsize": fs - 2, "ytick.labelsize": fs - 2,
                         "legend.fontsize": fs - 5, "figure.titlesize": fs + 2})

    records = scan(args.roots)
    ref = reference_final_step(records)
    per = {}
    for r in records:
        target = ref.get(r["ep"])
        cs = r["curves"].get(args.probe_file)
        if not cs or (target and cs[3] < target):
            continue
        for t, v in cs[4].items():
            per.setdefault((r["ep"], r["buf"]), {}).setdefault(t, []).append(v * 100)
    if not per:
        sys.exit("nothing to plot")

    budgets = sorted({e for e, _ in per})
    buffers = sorted({b for _, b in per})
    ers = [b for b in buffers if b > 0]
    cbufs = sorted(set(int(x) for x in args.color_buffers.split(",") if x) | set(ers))
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(1, len(budgets), figsize=(args.panel_width * len(budgets), args.panel_height), sharey=True)
    if len(budgets) == 1:
        axes = [axes]
    for ax, u in zip(axes, budgets):
        for b in buffers:
            d = per.get((u, b))
            if not d:
                continue
            ts = sorted(d)
            m = [mean(d[t]) for t in ts]
            if b == 0:
                ax.plot(ts, m, lw=2.0, ls="--", color=GRAY, marker="o", ms=4, label="Naive")
            else:
                ax.plot(ts, m, lw=1.6, color=cmap(cbufs.index(b) / max(len(cbufs) - 1, 1)),
                        marker="o", ms=3.5, label=f"ER {b}")
        ax.axhline(0, lw=.8, color="#999999")
        ax.set_xticks(sorted({t for (e, _), d in per.items() if e == u for t in d}))
        ax.set(xlabel=args.xlabel,
               title=f"{u} update" + ("s" if u > 1 else ""))
        ax.grid(alpha=.3)
    axes[0].set_ylabel(r"$F^{\mathrm{deep}}$ (pp)")
    if args.ylim:
        lo, hi = (float(x) for x in args.ylim.split(","))
        axes[0].set_ylim(lo, hi)
    def order_of(labels):
        return sorted(labels, key=lambda l: -1 if l == "Naive" else int(l.split()[1]))
    ncols_l = [int(x) for x in str(args.legend_ncols).split(",")]
    leg_kw = dict(frameon=False, fontsize=args.legend_fontsize,
                  handlelength=2.2, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.3)
    locs = args.legend_loc.split(",")
    if args.legend_in == "each":
        for i, ax in enumerate(axes):
            hl1 = dict((l, h) for h, l in zip(*ax.get_legend_handles_labels()))
            o = order_of(hl1)
            ax.legend([hl1[l] for l in o], o, loc=locs[min(i, len(locs) - 1)],
                      ncols=ncols_l[min(i, len(ncols_l) - 1)], **leg_kw)
    leg_ax = axes[0] if args.legend_in == "first" else axes[-1]
    # legend lists every buffer plotted in ANY panel, with the same colours
    hl = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(l, h)
    if args.legend_in != "each":
        order = order_of(hl)
        leg_ax.legend([hl[l] for l in order], order, loc=locs[0], ncols=ncols_l[0], **leg_kw)
    probe = "SGD" if args.probe_file == "lp_curve_sgd.csv" else "logreg"
    if not args.no_suptitle:
        fig.suptitle(f"Deep forgetting by task position and buffer size (probe: {probe})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight")
        print(f"-> {args.out}.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
