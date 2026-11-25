#!/usr/bin/env python3
"""Full cross-epoch analysis of the 1/3/5-epoch sweep: master table + figures.

Walks one or more result roots (results_ep1, results_ep3, results_ep5), reads
every run's logs.json / lp_curve.csv / lp_curve_sgd.csv / snr_curve.csv,
aggregates over seeds (mean +/- std) and writes

  summary_table.csv / summary_table.md
      every metric x every buffer x every epoch setting:
        Top1  = final average output accuracy            (%)
        LP    = final average linear-probe accuracy      (%)  [LogisticRegression]
        LPsgd = final average linear-probe accuracy      (%)  [SGDClassifier]
        Fsh   = shallow forgetting  (learned - final)    (pp)
        Fdeep = deep forgetting via LP                   (pp)
        FdeepSGD = deep forgetting via SGD probe         (pp)
        SNR   = final signal-to-noise ratio              (ratio)

  epochs_forgetting.png   shallow & deep forgetting vs buffer, one line per epoch
  epochs_accuracy.png     Top1 & LP accuracy vs buffer, one line per epoch
  epochs_snr.png          final SNR vs buffer, one line per epoch
  probe_logreg_vs_sgd.png LogisticRegression vs SGDClassifier probe: accuracy and
                          the deep-forgetting number each of them yields
"""
import argparse
import csv
import glob
import json
import os
import re
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = re.compile(r"Top1_Acc_Exp/eval_phase/test_stream/Task0*\d+/Exp0*(\d+)$")

BLUE, ORANGE, GRAY = "#1f77b4", "#d95f02", "#666666"
EPOCH_STYLE = {  # colour + marker per epoch setting (CVD-safe trio)
    1: ("#1f77b4", "o"),
    3: ("#d95f02", "s"),
    5: ("#2ca02c", "^"),
    20: ("#7570b3", "D"),
    50: ("#e7298a", "v"),
}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def parse_buffer(folder):
    """er_split_cifar100_10_2000 -> 2000 ; naive_... -> 0."""
    if folder.lower().startswith("naive"):
        return 0
    m = re.search(r"(\d+)$", folder)
    return int(m.group(1)) if m else None


def read_epochs(run_dir, root):
    """train_epochs from config.yaml, falling back to the root name (…_ep3)."""
    cfg = os.path.join(run_dir, "config.yaml")
    if os.path.isfile(cfg):
        m = re.search(r"train_epochs:\s*(\d+)", open(cfg).read())
        if m:
            return int(m.group(1))
    m = re.search(r"ep(\d+)", os.path.basename(os.path.normpath(root)))
    return int(m.group(1)) if m else None


def shallow_stats(logs_json):
    """(final avg Top-1, avg shallow forgetting) from the anytime log."""
    ckpts = []
    with open(logs_json) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            accs = {int(m.group(1)): v for k, v in d.items()
                    for m in [EXP.match(k)] if m}
            if accs:
                ckpts.append(accs)
    if not ckpts:
        return None
    last = {}
    for a in ckpts:
        last[max(a)] = a
    n = max(last) + 1
    R = [[None] * n for _ in range(n)]
    for j, a in last.items():
        for i, v in a.items():
            R[j][i] = v
    if any(R[n - 1][i] is None or R[i][i] is None for i in range(n)):
        return None  # incomplete run: don't silently report a wrong number
    final = [R[n - 1][i] for i in range(n)]
    fg = [R[i][i] - R[n - 1][i] for i in range(n - 1)]
    return mean(final), mean(fg)


def curve_stats(csv_path, col):
    """(curve {step: mean-over-tasks}, value at final step, deep forgetting, last_step).

    'learned' for task t is the last checkpoint at which t is the most recently
    trained task -- the same boundary detection compute_lp.py uses.

    NOTE: the caller MUST check last_step against the run's true end of training.
    Forgetting is learned-minus-final, so on a curve that stops early "final" is
    silently taken mid-stream and the number becomes meaningless.
    """
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["step"]), int(r["task"]), float(r[col])))
    if not rows:
        return None
    by_step = {}
    for s, _, v in rows:
        by_step.setdefault(s, []).append(v)
    curve = {s: mean(v) for s, v in sorted(by_step.items())}
    last = max(by_step)
    acc = {(s, t): v for s, t, v in rows}
    tasks = sorted({t for _, t, _ in rows})
    fg = []
    per_task = {}  # task -> forgetting, kept so the per-task figure can use it
    for t in tasks[:-1]:
        tsteps = [s for (s, tt) in acc if tt == t]
        newest = [s for s in tsteps
                  if max(x for (ss, x) in acc if ss == s) == t]
        learned = acc[(max(newest), t)] if newest else acc[(min(tsteps), t)]
        if (last, t) in acc:
            d = learned - acc[(last, t)]
            fg.append(d)
            per_task[t] = d
    return curve, curve[last], (mean(fg) if fg else 0.0), last, per_task


CURVE_FILES = [  # (filename, csv column, key for final value, key for forgetting, key for curve)
    ("lp_curve.csv", "probe_acc", "lp", "deep", "lp_curve", 100),
    ("lp_curve_sgd.csv", "probe_acc", "lpsgd", "deepsgd", "lpsgd_curve", 100),
    ("snr_curve.csv", "snr", "snr", None, "snr_curve", 1),
]

# Sparse end-of-task-only probe runs (compute_lp.py --boundaries-only). They carry
# the exact same 'learned' and 'final' points deep forgetting is built from, so the
# summary number is identical -- but they are NOT a dense curve, so they feed the
# table only and never the LP-over-training plots.
BOUNDARY_FILE = "lp_boundaries.csv"


def scan(roots):
    """One record per run, with each curve's last step kept for validation."""
    records = []
    for root in roots:
        for run_dir in sorted(glob.glob(os.path.join(root, "*", "[0-9]"))):
            folder = os.path.basename(os.path.dirname(run_dir))
            buf = parse_buffer(folder)
            ep = read_epochs(run_dir, root)
            if buf is None or ep is None:
                print(f"[skip] {run_dir}: cannot infer buffer/epochs")
                continue
            logs = os.path.join(run_dir, "logs.json")
            if not os.path.isfile(logs):
                print(f"[skip] {run_dir}: no logs.json")
                continue
            sh = shallow_stats(logs)
            if sh is None:
                print(f"[skip] {run_dir}: incomplete logs.json")
                continue

            rec = {"dir": run_dir, "ep": ep, "buf": buf,
                   "top1": sh[0] * 100, "shallow": sh[1] * 100, "curves": {}}
            for fname, col, _, _, _, _ in CURVE_FILES:
                path = os.path.join(run_dir, fname)
                rec["curves"][fname] = curve_stats(path, col) if os.path.isfile(path) else None
            bpath = os.path.join(run_dir, BOUNDARY_FILE)
            rec["curves"][BOUNDARY_FILE] = curve_stats(bpath, "probe_acc") \
                if os.path.isfile(bpath) else None
            records.append(rec)
    return records


def reference_final_step(records):
    """Per epoch setting, the true end-of-training step.

    Taken as the furthest step any curve of that setting reaches: a run that
    stops earlier was cut short (job timed out mid-analysis), not finished.
    """
    ref = {}
    for r in records:
        for cs in r["curves"].values():
            if cs:
                ref[r["ep"]] = max(ref.get(r["ep"], 0), cs[3])
    return ref


def build(records, ref):
    """Aggregate into {(epochs, buffer): {metric: [per-seed values]}}, dropping
    curves that stop before the end of training. Returns (data, incomplete)."""
    data = {}
    incomplete = {}  # (ep, fname) -> [(run, pct)]
    for r in records:
        e = data.setdefault((r["ep"], r["buf"]),
                            {k: [] for k in ("top1", "shallow", "lp", "deep",
                                             "lpsgd", "deepsgd", "snr",
                                             "lp_curve", "snr_curve", "lpsgd_curve")})
        e["top1"].append(r["top1"])
        e["shallow"].append(r["shallow"])

        target = ref.get(r["ep"])
        for fname, _, k_final, k_fg, k_curve, scale in CURVE_FILES:
            cs = r["curves"].get(fname)
            usable = cs and (not target or cs[3] >= target)

            # LogisticRegression fallback: if the dense curve is truncated but a
            # boundaries-only run reached the end, it still yields the exact same
            # deep-forgetting number (same learned/final points) -- just no curve.
            if fname == "lp_curve.csv" and not usable:
                b = r["curves"].get(BOUNDARY_FILE)
                if b and (not target or b[3] >= target):
                    e[k_final].append(b[1] * scale)
                    e[k_fg].append(b[2] * scale)
                    continue

            if not cs:
                continue
            if not usable:
                incomplete.setdefault((r["ep"], fname), []).append(
                    (os.sep.join(r["dir"].split(os.sep)[-2:]), 100 * cs[3] / target))
                continue  # truncated: 'final' would be mid-stream -> unusable
            curve, final_val, fg, _, _ = cs
            e[k_final].append(final_val * scale)
            if k_fg:
                e[k_fg].append(fg * scale)
            e[k_curve].append(curve)
    return data, incomplete


def report_incomplete(incomplete, ref):
    if not incomplete:
        print("All curves reach the end of training.\n")
        return
    print("\n" + "=" * 74)
    print("WARNING -- truncated curves excluded from the aggregates")
    print("=" * 74)
    print("Forgetting is learned-minus-final: on a curve that stops early the\n"
          "'final' point would be taken mid-stream, so these runs are dropped\n"
          "rather than reported as a wrong number. Finish them and re-run.\n")
    for (ep, fname), runs in sorted(incomplete.items()):
        pcts = sorted(p for _, p in runs)
        print(f"  ep{ep}  {fname:20s} {len(runs):2d} run(s) incomplete "
              f"(reach {pcts[0]:.0f}%-{pcts[-1]:.0f}% of step {ref.get(ep)})")
    print()


def ms(vals):
    if not vals:
        return None, None
    return mean(vals), (stdev(vals) if len(vals) > 1 else 0.0)


def fmt(vals, prec=1):
    """mean±std, unsigned (accuracies, SNR)."""
    m, s = ms(vals)
    return "-" if m is None else f"{m:.{prec}f}±{s:.{prec}f}"


def fmt_signed(vals, prec=1):
    """mean±std with an explicit sign -- forgetting can be negative
    (backward transfer), and the sign is itself informative."""
    m, s = ms(vals)
    return "-" if m is None else f"{m:+.{prec}f}±{s:.{prec}f}"


# --------------------------------------------------------------------------- #
# table
# --------------------------------------------------------------------------- #
COLUMNS = [
    ("Top1 (%)", "top1", fmt, 1),
    ("LP (%)", "lp", fmt, 1),
    ("LP-SGD (%)", "lpsgd", fmt, 1),
    ("F shallow (pp)", "shallow", fmt_signed, 1),
    ("F deep (pp)", "deep", fmt_signed, 1),
    ("F deep SGD (pp)", "deepsgd", fmt_signed, 1),
    ("SNR", "snr", fmt, 3),
]


def buffer_label(b):
    return "Naive (0)" if b == 0 else str(b)


def write_table(data, outdir):
    epochs = sorted({ep for ep, _ in data})
    buffers = sorted({b for _, b in data})

    csv_path = os.path.join(outdir, "summary_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epochs", "buffer", "n_seeds"]
                   + [c for c, _, _, _ in COLUMNS])
        for ep in epochs:
            for b in buffers:
                e = data.get((ep, b))
                if not e:
                    continue
                w.writerow([ep, b, len(e["top1"])]
                           + [fn(e[key], p) for _, key, fn, p in COLUMNS])
    print(f"-> {csv_path}")

    md_path = os.path.join(outdir, "summary_table.md")
    with open(md_path, "w") as f:
        f.write("# Cross-epoch summary (mean ± std over seeds)\n\n")
        f.write("Accuracies in %, forgetting in percentage points (pp), "
                "SNR dimensionless.\nForgetting is learned-minus-final, "
                "averaged over tasks 0..T-2.\n")
        for ep in epochs:
            f.write(f"\n## train_epochs = {ep}\n\n")
            f.write("| Buffer | " + " | ".join(c for c, _, _, _ in COLUMNS) + " |\n")
            f.write("|" + "---|" * (len(COLUMNS) + 1) + "\n")
            for b in buffers:
                e = data.get((ep, b))
                if not e:
                    continue
                cells = [fn(e[key], p) for _, key, fn, p in COLUMNS]
                f.write(f"| {buffer_label(b)} | " + " | ".join(cells) + " |\n")
    print(f"-> {md_path}")


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def buffer_axis(ax, buffers, naive_x):
    ax.set_xscale("log")
    ticks = ([naive_x] if 0 in buffers else []) + [b for b in buffers if b > 0]
    labels = (["0"] if 0 in buffers else []) + [str(b) for b in buffers if b > 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.minorticks_off()
    if 0 in buffers:
        ers = [b for b in buffers if b > 0]
        ax.axvline((naive_x * ers[0]) ** 0.5, color="#bbbbbb", lw=.8, ls=":")
    ax.grid(alpha=.3)


def series(data, ep, buffers, key, naive_x):
    xs, m, s = [], [], []
    for b in buffers:
        e = data.get((ep, b))
        if not e or not e[key]:
            continue
        mm, ss = ms(e[key])
        xs.append(naive_x if b == 0 else b)
        m.append(mm)
        s.append(ss)
    return xs, m, s


def plot_metric_panels(data, outdir, panels, fname, suptitle):
    """panels: list of (metric_key, ylabel, title)."""
    epochs = sorted({ep for ep, _ in data})
    buffers = sorted({b for _, b in data})
    ers = [b for b in buffers if b > 0]
    if not ers:
        return
    naive_x = ers[0] * 0.45

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.6))
    if len(panels) == 1:
        axes = [axes]
    for ax, (key, ylabel, title) in zip(axes, panels):
        drew = False
        for ep in epochs:
            col, mk = EPOCH_STYLE.get(ep, (GRAY, "o"))
            xs, m, s = series(data, ep, buffers, key, naive_x)
            if not xs:
                continue
            ax.errorbar(xs, m, yerr=s, marker=mk, ms=6, lw=1.8, capsize=3,
                        color=col, label=f"{ep} epoch" + ("s" if ep > 1 else ""))
            drew = True
        if not drew:
            ax.text(.5, .5, f"no data for '{key}'", ha="center", va="center",
                    transform=ax.transAxes, color=GRAY)
        if key.startswith(("shallow", "deep")):
            ax.axhline(0, lw=.8, color="#999999")
        buffer_axis(ax, buffers, naive_x)
        ax.set(xlabel="buffer size (0 = Naive)", ylabel=ylabel, title=title)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(suptitle)
    fig.tight_layout()
    out = os.path.join(outdir, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


def plot_probe_comparison(data, outdir):
    """LogisticRegression vs SGDClassifier: probe accuracy and deep forgetting."""
    epochs = sorted({ep for ep, _ in data})
    buffers = sorted({b for _, b in data})
    ers = [b for b in buffers if b > 0]
    if not ers:
        return
    naive_x = ers[0] * 0.45

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    # panel 1+2: accuracy and deep forgetting, solid = logreg, dashed = SGD
    for ax, (k_lr, k_sgd, ylabel, title) in zip(
            axes[:2],
            [("lp", "lpsgd", "final probe accuracy (%)",
              "Probe accuracy: LogisticRegression vs SGD"),
             ("deep", "deepsgd", "deep forgetting (pp)",
              "Deep forgetting: LogisticRegression vs SGD")]):
        for ep in epochs:
            col, mk = EPOCH_STYLE.get(ep, (GRAY, "o"))
            xs, m, s = series(data, ep, buffers, k_lr, naive_x)
            if xs:
                ax.errorbar(xs, m, yerr=s, marker=mk, ms=5, lw=1.8, capsize=2,
                            color=col, label=f"{ep}ep logreg")
            xs, m, s = series(data, ep, buffers, k_sgd, naive_x)
            if xs:
                ax.errorbar(xs, m, yerr=s, marker=mk, ms=5, lw=1.5, capsize=2,
                            color=col, ls="--", alpha=.75, label=f"{ep}ep SGD")
        if k_lr == "deep":
            ax.axhline(0, lw=.8, color="#999999")
        buffer_axis(ax, buffers, naive_x)
        ax.set(xlabel="buffer size (0 = Naive)", ylabel=ylabel, title=title)
        ax.legend(frameon=False, fontsize=7, ncols=2)

    # panel 3: direct scatter, one point per (epoch, buffer)
    ax = axes[2]
    pts_x, pts_y = [], []
    for ep in epochs:
        col, mk = EPOCH_STYLE.get(ep, (GRAY, "o"))
        xs, ys = [], []
        for b in buffers:
            e = data.get((ep, b))
            if not e or not e["lp"] or not e["lpsgd"]:
                continue
            xs.append(ms(e["lp"])[0])
            ys.append(ms(e["lpsgd"])[0])
        if xs:
            ax.scatter(xs, ys, marker=mk, color=col, s=42,
                       label=f"{ep} epoch" + ("s" if ep > 1 else ""))
            pts_x += xs
            pts_y += ys
    if pts_x:
        lo = min(pts_x + pts_y) - 2
        hi = max(pts_x + pts_y) + 2
        ax.plot([lo, hi], [lo, hi], color=GRAY, lw=.9, ls=":", label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
    ax.set(xlabel="LP accuracy, LogisticRegression (%)",
           ylabel="LP accuracy, SGDClassifier (%)",
           title="Agreement between the two probes")
    ax.grid(alpha=.3)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Linear probe: LogisticRegression (L-BFGS) vs SGDClassifier")
    fig.tight_layout()
    out = os.path.join(outdir, "probe_logreg_vs_sgd.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


def plot_curves_by_epoch(data, outdir, lp_curve_key="lp_curve"):
    """LP and SNR trajectories over training, one panel per epoch setting."""
    epochs = sorted({ep for ep, _ in data})
    buffers = sorted({b for _, b in data})

    for key, ylabel, fname in [(lp_curve_key, "linear-probe accuracy", "epochs_lp_curves.png"),
                               ("snr_curve", "SNR", "epochs_snr_curves.png")]:
        fig, axes = plt.subplots(1, len(epochs), figsize=(6.2 * len(epochs), 4.4),
                                 sharey=True)
        if len(epochs) == 1:
            axes = [axes]
        cmap = plt.get_cmap("viridis")
        ers = [b for b in buffers if b > 0]
        for ax, ep in zip(axes, epochs):
            for b in buffers:
                e = data.get((ep, b))
                if not e or not e[key]:
                    continue
                curves = e[key]
                steps = sorted(curves[0])
                common = [s for s in steps if all(s in c for c in curves)]
                if not common:
                    continue
                m = [mean([c[s] for c in curves]) for s in common]
                if b == 0:
                    ax.plot(common, m, lw=1.8, ls="--", color=GRAY, label="Naive")
                else:
                    col = cmap(ers.index(b) / max(len(ers) - 1, 1))
                    ax.plot(common, m, lw=1.6, color=col, label=f"ER {b}")
            ax.set(xlabel="training step", title=f"{ep} epoch" + ("s" if ep > 1 else ""))
            ax.grid(alpha=.3)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(frameon=False, fontsize=7, ncols=2)
        fig.suptitle(f"{ylabel} over training (mean over seeds)")
        fig.tight_layout()
        out = os.path.join(outdir, fname)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"-> {out}")


def task_boundary_steps(csv_path):
    """{task_id: step} -- the last step at which task_id is the newest task
    recorded, i.e. right after that task's training finished. Same boundary
    logic as plot_task_heatmap.csv_matrix, but returns the steps themselves
    instead of a value matrix.
    """
    by_step = {}
    with open(csv_path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            s, t = int(row[0]), int(row[1])
            by_step.setdefault(s, set()).add(t)
    if not by_step:
        return {}
    n = max(max(tasks) for tasks in by_step.values()) + 1
    out = {}
    for j in range(n):
        cand = [s for s, tasks in by_step.items() if max(tasks) == j]
        if cand:
            out[j] = max(cand)
    return out


def plot_snr_task_boundaries(data, records, outdir,
                             fname="epochs_snr_task_boundaries.png",
                             ub_file="upper_bound_curve.csv", probe_label=None):
    """SNR over training, split by task, vs. an offline upper-bound classifier.

    Requested by the thesis supervisors: same panels as epochs_snr_curves.png,
    with a vertical line at every task boundary, plus a step curve of an OFFLINE
    upper-bound classifier -- one linear model fit jointly on every class seen so
    far (not per-task, not online). Expected to decrease with task index for a
    purely combinatorial reason (more competing classes is harder even for a
    perfectly-fit classifier), independent of forgetting.
    """
    epochs = sorted({ep for ep, _ in data})
    by_dirs = {}
    for r in records:
        by_dirs.setdefault((r["ep"], r["buf"]), []).append(r["dir"])

    fig, axes = plt.subplots(1, len(epochs), figsize=(6.6 * len(epochs), 4.6), sharey=True)
    if len(epochs) == 1:
        axes = [axes]
    cmap = plt.get_cmap("viridis")
    twins, ub_vals = [], []
    handles = labels = None

    for ax, ep in zip(axes, epochs):
        buffers = sorted({b for e, b in data if e == ep})
        ers = [b for b in buffers if b > 0]

        for b in buffers:
            e = data.get((ep, b))
            curves = e["snr_curve"] if e else None
            if not curves:
                continue
            steps = sorted(curves[0])
            common = [s for s in steps if all(s in c for c in curves)]
            if not common:
                continue
            m = [mean([c[s] for c in curves]) for s in common]
            if b == 0:
                ax.plot(common, m, lw=1.8, ls="--", color=GRAY, label="Naive")
            else:
                col = cmap(ers.index(b) / max(len(ers) - 1, 1))
                ax.plot(common, m, lw=1.4, color=col, alpha=.85, label=f"ER {b}")

        # task boundaries + upper-bound accuracy, from the largest buffer's seeds
        ref_buf = max(buffers) if buffers else None
        dirs = by_dirs.get((ep, ref_buf), []) if ref_buf is not None else []

        boundaries = {}
        for d in dirs:
            csvp = os.path.join(d, "snr_curve.csv")
            if os.path.isfile(csvp):
                boundaries = task_boundary_steps(csvp)
                if boundaries:
                    break

        ub = {}
        for d in dirs:
            p = os.path.join(d, ub_file)
            if not os.path.isfile(p):
                continue
            with open(p) as f:
                for row in csv.DictReader(f):
                    ub.setdefault(int(row["boundary_task"]), []).append(
                        float(row["upper_bound_acc"]))

        ax2 = ax.twinx()
        twins.append(ax2)
        if ub and boundaries:
            xs, ys = [], []
            for j in sorted(ub):
                if j not in boundaries:
                    continue
                xs.append(boundaries[j])
                ys.append(mean(ub[j]) * 100)
            if xs:
                ax2.step(xs, ys, where="post", color="black", lw=2.0,
                         label=f"offline upper bound (ER {ref_buf})")
                ub_vals.extend(ys)

        for j, s in boundaries.items():
            if j > 0:
                ax.axvline(s, color="#8c564b", lw=1.1, alpha=.7, ls=":", zorder=0)

        ax.set(xlabel="training step", title=f"{ep} epoch" + ("s" if ep > 1 else ""))
        ax.grid(alpha=.3)
        if ax is axes[-1]:
            ax2.set_ylabel("offline upper-bound accuracy (%)")

        if handles is None:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            handles, labels = h1 + h2, l1 + l2

    # One shared right-hand scale: with a per-panel scale the three staircases
    # look identical, hiding the very thing this figure is for -- that more
    # epochs raise the ceiling the online run is measured against.
    if ub_vals:
        lo, hi = min(ub_vals), max(ub_vals)
        pad = max((hi - lo) * 0.08, 0.5)
        for ax2 in twins:
            ax2.set_ylim(lo - pad, hi + pad)

    axes[0].set_ylabel("SNR")
    if handles:
        fig.legend(handles, labels, frameon=False, fontsize=8, ncols=5,
                   loc="lower center", bbox_to_anchor=(0.5, -0.04))
    suffix = f" [{probe_label}]" if probe_label else ""
    fig.suptitle("SNR over training, split by task boundaries, vs. an offline "
                 f"upper-bound classifier{suffix} (mean over seeds)")
    fig.tight_layout()
    out = os.path.join(outdir, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


def plot_per_task(records, outdir, ref):
    """Deep forgetting as a function of TASK POSITION, one line per epoch setting.

    The aggregate number averages over a steep gradient: task 0 is learned against
    an almost-random backbone (so it keeps improving afterwards -> large negative
    forgetting) while the last tasks are learned against a mature one. Collapsing
    that into a single mean hides the effect entirely, which is why this panel
    exists alongside the buffer-axis figures.
    """
    curves = {}
    for r in records:
        target = ref.get(r["ep"])
        for fname, key in (("lp_curve.csv", "logreg"), ("lp_curve_sgd.csv", "sgd")):
            cs = r["curves"].get(fname)
            if not cs or (target and cs[3] < target):
                continue
            for t, v in cs[4].items():
                curves.setdefault((key, r["ep"]), {}).setdefault(t, []).append(v * 100)

    probes = sorted({k for k, _ in curves})
    if not probes:
        return
    fig, axes = plt.subplots(1, len(probes), figsize=(6.4 * len(probes), 4.6), sharey=True)
    if len(probes) == 1:
        axes = [axes]
    for ax, probe in zip(axes, probes):
        for ep in sorted({e for k, e in curves if k == probe}):
            per_task = curves[(probe, ep)]
            ts = sorted(per_task)
            m = [mean(per_task[t]) for t in ts]
            s = [ms(per_task[t])[1] for t in ts]
            col, mk = EPOCH_STYLE.get(ep, (GRAY, "o"))
            ax.errorbar(ts, m, yerr=s, marker=mk, ms=6, lw=1.8, capsize=3,
                        color=col, label=f"{ep} epoch" + ("s" if ep > 1 else ""))
        ax.axhline(0, lw=.8, color="#999999")
        ax.set(xlabel="task index (order in the stream)",
               title=f"probe: {probe}")
        ax.grid(alpha=.3)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("deep forgetting (pp)")
    fig.suptitle("Deep forgetting by task position — negative = backward transfer")
    fig.tight_layout()
    out = os.path.join(outdir, "per_task_forgetting.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


def plot_per_task_by_buffer(records, outdir, ref, probe_file="lp_curve.csv",
                            fname="per_task_by_buffer.png"):
    """Per-task deep forgetting resolved by buffer size, one panel per epoch setting.

    plot_per_task() averages over every buffer, which answers "how does forgetting
    depend on task position" but hides whether replay changes the shape of that
    dependence. Here each buffer gets its own curve, so the two factors -- position
    in the stream and memory size -- can be read off at once.
    """
    per = {}  # (ep, buf) -> {task: [values]}
    for r in records:
        target = ref.get(r["ep"])
        cs = r["curves"].get(probe_file)
        if not cs or (target and cs[3] < target):
            continue
        for t, v in cs[4].items():
            per.setdefault((r["ep"], r["buf"]), {}).setdefault(t, []).append(v * 100)
    if not per:
        return

    epochs = sorted({e for e, _ in per})
    buffers = sorted({b for _, b in per})
    ers = [b for b in buffers if b > 0]
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(1, len(epochs), figsize=(5.8 * len(epochs), 4.6), sharey=True)
    if len(epochs) == 1:
        axes = [axes]
    for ax, ep in zip(axes, epochs):
        for b in buffers:
            d = per.get((ep, b))
            if not d:
                continue
            ts = sorted(d)
            m = [mean(d[t]) for t in ts]
            if b == 0:
                ax.plot(ts, m, lw=2.0, ls="--", color=GRAY, marker="o", ms=4, label="Naive")
            else:
                ax.plot(ts, m, lw=1.6, color=cmap(ers.index(b) / max(len(ers) - 1, 1)),
                        marker="o", ms=3.5, label=f"ER {b}")
        ax.axhline(0, lw=.8, color="#999999")
        ax.set(xlabel="task index (order in the stream)",
               title=f"{ep} epoch" + ("s" if ep > 1 else ""))
        ax.grid(alpha=.3)
    axes[0].set_ylabel("deep forgetting (pp)")
    axes[-1].legend(frameon=False, fontsize=7, ncols=2)
    fig.suptitle("Deep forgetting by task position and buffer size "
                 f"(probe: {'logreg' if probe_file == 'lp_curve.csv' else 'SGD'})")
    fig.tight_layout()
    out = os.path.join(outdir, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="result roots, e.g. results_ep1 results_ep3 results_ep5")
    ap.add_argument("--outdir", default="analysis_out")
    ap.add_argument("--tolerance-steps", type=int, default=0,
                    help="accept curves ending up to N steps before the last checkpoint "
                         "(e.g. 200 for a stride-5 SNR curve: last point at 249750 vs 249950)")
    ap.add_argument("--probe", choices=["auto", "sgd"], default="auto",
                    help="sgd: every figure uses the SGDClassifier probe "
                         "(lp_curve_sgd.csv), even where lp_curve.csv exists")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    records = scan(args.roots)
    if not records:
        print("No runs found under:", args.roots)
        return
    ref = reference_final_step(records)
    if args.tolerance_steps:
        ref = {ep: max(0, v - args.tolerance_steps) for ep, v in ref.items()}
    data, incomplete = build(records, ref)
    report_incomplete(incomplete, ref)

    found = sorted({ep for ep, _ in data})
    print(f"Found epoch settings: {found}; "
          f"{len(data)} (epoch, buffer) cells\n")

    # If the LogisticRegression probe is missing (e.g. still being computed), fall
    # back to the SGD probe: it is the same metric measured with a different
    # classifier, so the figures stay informative instead of showing empty panels.
    has_lr = any(e["deep"] for e in data.values()) and args.probe != "sgd"
    deep_key, deep_lbl = ("deep", "linear probe") if has_lr else ("deepsgd", "linear probe, SGD")
    lp_key, lp_lbl = ("lp", "LP") if has_lr and any(e["lp"] for e in data.values()) else ("lpsgd", "LP-SGD")
    if not has_lr:
        print("\n[info] no LogisticRegression probe data -> figures use the SGD probe\n")

    write_table(data, args.outdir)
    plot_metric_panels(
        data, args.outdir,
        [("shallow", "shallow forgetting (pp)", "Shallow (output layer)"),
         (deep_key, "deep forgetting (pp)", f"Deep ({deep_lbl})")],
        "epochs_forgetting.png",
        "Forgetting vs buffer size, across epoch settings")
    plot_metric_panels(
        data, args.outdir,
        [("top1", "final Top-1 accuracy (%)", "Output accuracy"),
         (lp_key, f"final {lp_lbl} accuracy (%)", f"{lp_lbl} accuracy")],
        "epochs_accuracy.png",
        "Accuracy vs buffer size, across epoch settings")
    plot_metric_panels(
        data, args.outdir,
        [("snr", "final SNR", "Feature separability")],
        "epochs_snr.png",
        "SNR vs buffer size, across epoch settings")
    plot_probe_comparison(data, args.outdir)
    plot_curves_by_epoch(data, args.outdir,
                         "lpsgd_curve" if args.probe == "sgd" else "lp_curve")
    plot_snr_task_boundaries(data, records, args.outdir,
                             probe_label="LogisticRegression")
    plot_snr_task_boundaries(data, records, args.outdir,
                             fname="epochs_snr_task_boundaries_sgd.png",
                             ub_file="upper_bound_curve_sgd.csv",
                             probe_label="SGDClassifier")
    plot_per_task(records, args.outdir, ref)
    if args.probe != "sgd":
        plot_per_task_by_buffer(records, args.outdir, ref)
    plot_per_task_by_buffer(records, args.outdir, ref, "lp_curve_sgd.csv",
                            "per_task_by_buffer_sgd.png")

    print(f"\nAll outputs in: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
