#!/usr/bin/env python3
"""Figure-3-style Neural Collapse plots.

Reproduces the layout of Lanzillotta et al. Figure 3, TIL row

  NC1  optimum 0      variability collapse
  NC2  optimum -1/(K-1)   mean within-task pairwise cosine of centered class means
       (the cross-task cosine is overlaid in violet, as in the paper: in TIL it has
       no reason to reach the ETF value, since nothing in the loss couples the heads)
  NC3  optimum 1      self-duality between head weights and class means
"""
import argparse
import csv
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EPOCH_STYLE = {1: ("#1f77b4", "o"), 3: ("#d95f02", "s"), 5: ("#2ca02c", "^"),
               20: ("#7570b3", "D"), 50: ("#e7298a", "v")}
VIOLET, GREY = "#7b3294", "#888888"


def load(run_dir, fname="nc_curve.csv"):
    path = os.path.join(run_dir, fname)
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: (int(v) if k in ("step", "task") else float(v))
                         for k, v in r.items()})
    return rows or None


def epochs_of(run_dir):
    cfg = os.path.join(run_dir, "config.yaml")
    if os.path.isfile(cfg):
        m = re.search(r"train_epochs:\s*(\d+)", open(cfg).read())
        if m:
            return int(m.group(1))
    m = re.search(r"ep(\d+)", run_dir)
    return int(m.group(1)) if m else None


def n_classes(run_dir, default=10):
    cfg = os.path.join(run_dir, "config.yaml")
    if os.path.isfile(cfg):
        txt = open(cfg).read()
        m = re.search(r"n_experiences:\s*(\d+)", txt)
        if m:
            return 100 // int(m.group(1))
    return default


def per_task_panels(rows, K, title, out):
    """Figure-3 layout: one curve per task, three panels."""
    tasks = sorted({r["task"] for r in rows})
    steps = sorted({r["step"] for r in rows})
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    specs = [("nc1", r"$\mathcal{NC}1$", 0.0, "variability collapse"),
             ("nc2_cos_mean", r"$\mathcal{NC}2$", -1.0 / (K - 1), "within-task ETF"),
             ("nc3", r"$\mathcal{NC}3$", 1.0, "self-duality")]

    for ax, (key, lab, opt, sub) in zip(axes, specs):
        for t in tasks:
            pts = [(r["step"], r[key]) for r in rows if r["task"] == t]
            pts.sort()
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        lw=1.4, color=cmap(t % 10), label=f"T{t}")
        if key == "nc2_cos_mean":
            pts = sorted({(r["step"], r["cross_task_cos"]) for r in rows})
            pts = [p for p in pts if not np.isnan(p[1])]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        lw=1.6, color=VIOLET, ls="-", alpha=.85, label="between tasks")
        ax.axhline(opt, color="black", lw=1.0, ls="--", zorder=1)
        ax.annotate(f"optimum = {opt:g}", xy=(0.98, opt), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=8, color="black")
        # task switches
        for t in tasks[1:]:
            first = min(r["step"] for r in rows if r["task"] == t)
            ax.axvline(first, color="#8c564b", lw=.7, alpha=.5)
        ax.set(xlabel="training step", title=f"{lab}  --  {sub}")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("metric value")
    axes[-1].legend(frameon=False, fontsize=7, ncols=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


def compare_runs(runs, out):
    """One curve per run (mean over tasks), to compare epoch settings directly."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    K = n_classes(runs[0])
    specs = [("nc1", r"$\mathcal{NC}1$", 0.0),
             ("nc2_cos_mean", r"$\mathcal{NC}2$", -1.0 / (K - 1)),
             ("nc3", r"$\mathcal{NC}3$", 1.0)]

    for ax, (key, lab, opt) in zip(axes, specs):
        for run in runs:
            rows = load(run)
            if not rows:
                print(f"[skip] {run}: no nc_curve.csv")
                continue
            ep = epochs_of(run)
            col, mk = EPOCH_STYLE.get(ep, (GREY, "o"))
            by_step = {}
            for r in rows:
                by_step.setdefault(r["step"], []).append(r[key])
            xs = sorted(by_step)
            ys = [float(np.nanmean(by_step[s])) for s in xs]
            # normalise the x axis: the three regimes have different stream lengths,
            # so absolute steps are not comparable -- fraction of training is.
            xn = [s / max(xs) for s in xs]
            ax.plot(xn, ys, lw=1.6, color=col,
                    label=f"{ep} epoch" + ("s" if ep and ep > 1 else ""))
        ax.axhline(opt, color="black", lw=1.0, ls="--")
        ax.annotate(f"optimum = {opt:g}", xy=(0.98, opt), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=8)
        ax.set(xlabel="fraction of training", title=lab)
        ax.grid(alpha=.3)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("metric value (mean over tasks)")
    fig.suptitle("Neural Collapse across epoch settings")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


def paper_grid(runs, out, title=None, max_tasks=5):
    """The reference paper's Figure 3, transposed to this study.

    There the rows are CL scenarios (Class-IL / Domain-IL / Task-IL) and the
    columns are NC1/NC2/NC3. Every run here is Task-IL, so the rows become the
    epoch settings instead -- which turns the same layout into a direct answer to
    whether more updates per sample drive the geometry further towards collapse.

    Like their figure, only the first `max_tasks` tasks are drawn and the x axis
    stops at that task's end: their Table 1 benchmark is 10 tasks, exactly ours,
    and Figure 3 shows the first three of them for legibility rather than all.
    """
    rows = []
    for run in runs:
        r = load(run)
        if r:
            rows.append((epochs_of(run), r, n_classes(run)))
    if not rows:
        print("[skip] paper grid: no nc_curve.csv among the given runs")
        return
    rows.sort(key=lambda x: (x[0] is None, x[0]))

    specs = [("nc1", r"$\mathcal{NC}1$", 0.0, "Optimum = 0"),
             ("nc2_cos_mean", r"$\mathcal{NC}2$", None,
              r"Optimum = $-\frac{1}{K-1}$"),
             ("nc3", r"$\mathcal{NC}3$", 1.0, "Optimum = 1")]

    fig, axes = plt.subplots(len(rows), 3, figsize=(11.5, 2.35 * len(rows)),
                             squeeze=False)
    cmap = plt.get_cmap("tab10")
    box = dict(facecolor="#dff0d8", edgecolor="none", alpha=.85,
               boxstyle="round,pad=0.25")

    for i, (ep, data, K) in enumerate(rows):
        tasks = sorted({d["task"] for d in data})[:max_tasks]
        keep = set(tasks)
        starts = {t: min(d["step"] for d in data if d["task"] == t) for t in tasks}
        # stop where the last shown task ends: the next task's first checkpoint
        after = [d["step"] for d in data if d["task"] not in keep]
        xmax = min(after) if after else max(d["step"] for d in data)

        for j, (key, lab, opt, opt_txt) in enumerate(specs):
            ax = axes[i][j]
            target = -1.0 / (K - 1) if opt is None else opt

            for t in tasks:
                pts = sorted((d["step"], d[key]) for d in data
                             if d["task"] == t and d["step"] <= xmax)
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts],
                            lw=1.1, color=cmap(t % 10), label=f"T{t + 1}",
                            solid_joinstyle="round")

            if key == "nc2_cos_mean":
                pts = sorted({(d["step"], d["cross_task_cos"]) for d in data
                              if d["step"] <= xmax})
                pts = [p for p in pts if not np.isnan(p[1])]
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts],
                            lw=1.3, color=VIOLET, label="between tasks")

            ax.axhline(target, color="black", lw=.9, ls="--", zorder=1)
            # alternating bands make each task's stretch of the stream explicit:
            # task t owns [starts[t], starts[t+1]), the last one runs to xmax
            for n, t in enumerate(tasks):
                lo_x = starts[t]
                hi_x = starts[tasks[n + 1]] if n + 1 < len(tasks) else xmax
                if n % 2:
                    ax.axvspan(lo_x, hi_x, color="#000000", alpha=.035, zorder=0)
                if i == 0:
                    ax.annotate(f"T{t + 1}", xy=((lo_x + hi_x) / 2, 1.005),
                                xycoords=("data", "axes fraction"),
                                ha="center", va="bottom", fontsize=6.5,
                                color="#555555")
            for t in tasks[1:]:
                ax.axvline(starts[t], color="#8c564b", lw=.7, ls=":", alpha=.75,
                           zorder=0)

            ax.set_xlim(-xmax * .02, xmax * 1.02)
            ax.set_xticks([starts[t] for t in tasks] + [xmax])
            ax.tick_params(labelsize=7.5)
            ax.grid(alpha=.18, lw=.5)
            if i == 0:
                ax.set_title(lab, fontsize=13, pad=14)
            # every row keeps its own tick labels: the three epoch settings run for
            # different numbers of steps, so hiding them would imply a shared x axis
            # that does not exist
            if i == len(rows) - 1:
                ax.set_xlabel("training step", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{ep} epoch" + ("s" if ep and ep > 1 else ""),
                              fontsize=10.5)
            # the optimum callout goes on the middle row only, as in their figure
            if i == len(rows) // 2:
                ax.annotate(opt_txt, xy=(.97, .5), xycoords="axes fraction",
                            ha="right", va="center", fontsize=7.5, bbox=box)

    # the task colours and the violet curve mean the same thing in every panel,
    # so they are named once instead of in all nine
    axes[0][0].legend(frameon=False, fontsize=6.5, ncols=2, loc="upper right",
                      handlelength=1.2, columnspacing=.9, handletextpad=.4)
    h, l = axes[0][1].get_legend_handles_labels()
    if "between tasks" in l:
        k = l.index("between tasks")
        axes[0][1].legend([h[k]], [l[k]], frameon=False, fontsize=6.5,
                          loc="lower right", handlelength=1.2, handletextpad=.4)

    # one shared scale per column: a row-to-row difference must be a real
    # difference, not the axes rescaling under each panel
    for j in range(3):
        lo = min(axes[i][j].get_ylim()[0] for i in range(len(rows)))
        hi = max(axes[i][j].get_ylim()[1] for i in range(len(rows)))
        for i in range(len(rows)):
            axes[i][j].set_ylim(lo, hi)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--compare", action="store_true",
                    help="overlay the runs (mean over tasks) instead of one figure each")
    ap.add_argument("--paper-grid", action="store_true",
                    help="the reference paper's Figure 3 layout: one row per run "
                         "(epoch setting), columns NC1/NC2/NC3, one curve per task")
    ap.add_argument("--title", default=None)
    ap.add_argument("--max-tasks", type=int, default=5,
                    help="how many of the first tasks to draw (paper grid)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.paper_grid:
        paper_grid(args.runs, args.out or "nc_paper_grid.png", args.title,
                   args.max_tasks)
        return

    if args.compare:
        compare_runs(args.runs, args.out or "nc_compare.png")
        return

    for run in args.runs:
        rows = load(run)
        if not rows:
            print(f"[skip] {run}: no nc_curve.csv -- run compute_nc.py first")
            continue
        K = n_classes(run)
        name = os.sep.join(os.path.normpath(run).split(os.sep)[-3:])
        out = args.out or os.path.join(run, "nc_figure.png")
        per_task_panels(rows, K, f"Neural Collapse -- {name}", out)


if __name__ == "__main__":
    main()
