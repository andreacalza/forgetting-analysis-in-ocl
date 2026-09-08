#!/usr/bin/env python3
"""Neural Collapse summary tables: one table per epoch setting.

Each table has one row per replay budget (Naive + the seven ER sizes), reporting
mean +/- std over seeds of the end-of-training value, itself averaged over every
task's head. Columns are the three metrics of Lanzillotta et al.'s Figure 3 plus
the cross-task cosine, which is the multi-head-specific one (their Proposition 1:
nothing in the loss couples separate heads, so it has no reason to reach the ETF
value the within-task cosine converges to).

Usage:
    python scripts/analysis/make_nc_tables.py results_ep1 results_ep3 results_ep5
    python scripts/analysis/make_nc_tables.py results_ep* --format latex
"""
import argparse
import csv
import glob
import os
import re
import sys
from statistics import mean, pstdev

BUFFERS = [(0, "Naive"), (200, "ER 200"), (500, "ER 500"), (1000, "ER 1000"),
           (2000, "ER 2000"), (5000, "ER 5000"), (10000, "ER 10000"),
           (50000, "ER 50000")]

COLS = [("nc1", r"$\mathcal{NC}1$", "{:.2f}", "{:.2f}"),
        ("nc2_cos_mean", r"$\mathcal{NC}2$", "{:+.4f}", "{:.4f}"),
        ("nc3", r"$\mathcal{NC}3$", "{:.3f}", "{:.3f}"),
        ("cross_task_cos", "cross-task", "{:+.4f}", "{:.4f}")]

PLAIN = {r"$\mathcal{NC}1$": "NC1", r"$\mathcal{NC}2$": "NC2 cos",
         r"$\mathcal{NC}3$": "NC3", "cross-task": "cross-task"}


def parse_buffer(folder):
    if folder.startswith("naive"):
        return 0
    m = re.search(r"_(\d+)$", folder)
    return int(m.group(1)) if m else None


def epochs_of(run_dir, root):
    cfg = os.path.join(run_dir, "config.yaml")
    if os.path.isfile(cfg):
        m = re.search(r"train_epochs:\s*(\d+)", open(cfg).read())
        if m:
            return int(m.group(1))
    m = re.search(r"ep(\d+)", root)
    return int(m.group(1)) if m else None


def final_values(path):
    """Per-metric mean over tasks at the last checkpoint of one run."""
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    last = max(int(r["step"]) for r in rows)
    sel = [r for r in rows if int(r["step"]) == last]
    out = {}
    for key, _, _, _ in COLS:
        vals = [float(r[key]) for r in sel
                if r.get(key) not in (None, "") and r[key].lower() != "nan"]
        out[key] = mean(vals) if vals else None
    return out


def collect(roots):
    """{(epochs, buffer): {metric: [one value per seed]}}"""
    data = {}
    for root in roots:
        for run_dir in sorted(glob.glob(os.path.join(root, "*", "[0-9]"))):
            path = os.path.join(run_dir, "nc_curve.csv")
            if not os.path.isfile(path):
                continue
            buf = parse_buffer(os.path.basename(os.path.dirname(run_dir)))
            ep = epochs_of(run_dir, root)
            if buf is None or ep is None:
                continue
            fv = final_values(path)
            if not fv:
                continue
            cell = data.setdefault((ep, buf), {k: [] for k, _, _, _ in COLS})
            for key, _, _, _ in COLS:
                if fv[key] is not None:
                    cell[key].append(fv[key])
    return data


def emit_csv(data, ep, path):
    """One CSV per epoch setting: a row per buffer, mean and std in separate columns
    so the numbers stay machine-readable instead of being baked into a string."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        head = ["buffer", "n_seeds"]
        for key, _, _, _ in COLS:
            head += [key, key + "_std"]
        w.writerow(head)
        for buf, label in BUFFERS:
            cell = data.get((ep, buf))
            if not cell:
                continue
            n = max((len(v) for v in cell.values()), default=0)
            row = [label, n]
            for key, _, _, _ in COLS:
                v = cell[key]
                row += ["" if not v else mean(v),
                        "" if len(v) < 2 else pstdev(v)]
            w.writerow(row)
    print(f"-> {path}", file=sys.stderr)


def emit_text(data, ep, out):
    """Aligned plain-text table, the same shape as the CSV."""
    print(f"===== {ep} epoch{'s' if ep > 1 else ''} "
          f"(mean +/- std over seeds, averaged over task heads) =====", file=out)
    print(f"{'buffer':<10}" + "".join(f"{PLAIN[h]:>22}" for _, h, _, _ in COLS),
          file=out)
    print("-" * (10 + 22 * len(COLS)), file=out)
    for buf, label in BUFFERS:
        cell = data.get((ep, buf))
        if not cell:
            continue
        line = f"{label:<10}"
        for key, _, fm, fs in COLS:
            v = cell[key]
            if not v:
                line += f"{'--':>22}"
            elif len(v) > 1:
                line += f"{fm.format(mean(v)) + ' +/- ' + fs.format(pstdev(v)):>22}"
            else:
                line += f"{fm.format(v[0]):>22}"
        print(line, file=out)
    print("\noptima: NC1 -> 0 | NC2 -> -0.1111 | NC3 -> 1 | cross-task: no target\n",
          file=out)


def emit_latex(data, ep, out):
    seeds = max((len(v) for (e, _), c in data.items() if e == ep
                 for v in c.values() if v), default=0)
    print(r"\begin{table}[H]", file=out)
    print(r"\centering", file=out)
    print(rf"\caption{{Neural Collapse metrics at the end of training, "
          rf"{ep} epoch{'s' if ep > 1 else ''} "
          rf"({seeds}-seed mean $\pm$ std, averaged over task heads). "
          rf"Optima: $\mathcal{{NC}}1 \to 0$, $\mathcal{{NC}}2 \to -1/(K-1) = -0.1111$, "
          rf"$\mathcal{{NC}}3 \to 1$. The cross-task cosine has no such target: "
          rf"separate heads are geometrically unconstrained.}}", file=out)
    print(rf"\label{{tab:nc-ep{ep}}}", file=out)
    print(r"\small", file=out)
    print(r"\begin{tabular}{l" + "r" * len(COLS) + "}", file=out)
    print(r"\toprule", file=out)
    print("Buffer & " + " & ".join(h for _, h, _, _ in COLS) + r" \\", file=out)
    print(r"\midrule", file=out)
    for buf, label in BUFFERS:
        cell = data.get((ep, buf))
        if not cell:
            continue
        parts = []
        for key, _, fm, fs in COLS:
            v = cell[key]
            if not v:
                parts.append("--")
            elif len(v) > 1:
                parts.append(f"${fm.format(mean(v))} \\pm {fs.format(pstdev(v))}$")
            else:
                parts.append(f"${fm.format(v[0])}$")
        print(f"{label:<9} & " + " & ".join(parts) + r" \\", file=out)
    print(r"\bottomrule", file=out)
    print(r"\end{tabular}", file=out)
    print(r"\end{table}", file=out)
    print("", file=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--format", choices=["csv", "text", "latex"], default="csv",
                    help="csv: one nc_table_ep<N>.csv per epoch setting (default); "
                         "text: aligned tables on stdout; latex: table environments")
    ap.add_argument("--outdir", default=".", help="where the CSVs go (--format csv)")
    ap.add_argument("--out", default=None,
                    help="single output file for --format text/latex")
    args = ap.parse_args()

    data = collect(args.roots)
    if not data:
        print("No nc_curve.csv found under:", args.roots, file=sys.stderr)
        return

    epochs = sorted({e for e, _ in data})

    if args.format == "csv":
        os.makedirs(args.outdir, exist_ok=True)
        for ep in epochs:
            emit_csv(data, ep, os.path.join(args.outdir, f"nc_table_ep{ep}.csv"))
        for ep in epochs:
            emit_text(data, ep, sys.stdout)
        return

    out = open(args.out, "w") if args.out else sys.stdout
    emit = emit_text if args.format == "text" else emit_latex
    for ep in epochs:
        emit(data, ep, out)
    if args.out:
        out.close()
        print(f"-> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
