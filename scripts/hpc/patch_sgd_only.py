#!/usr/bin/env python3

import sys


def patch(path, old, new):
    s = open(path).read()
    if new in s:
        return
    if old not in s:
        sys.exit(f"PATCH FAILED in {path}: expected text not found")
    open(path, "w").write(s.replace(old, new, 1))


A = "scripts/analysis/analyze_epochs.py"
patch(A, '    5: ("#2ca02c", "^"),\n}',
         '    5: ("#2ca02c", "^"),\n    20: ("#7570b3", "D"),\n    50: ("#e7298a", "v"),\n}')
patch(A, 'def plot_curves_by_epoch(data, outdir):',
         'def plot_curves_by_epoch(data, outdir, lp_curve_key="lp_curve"):')
patch(A, '[("lp_curve", "linear-probe accuracy", "epochs_lp_curves.png"),',
         '[(lp_curve_key, "linear-probe accuracy", "epochs_lp_curves.png"),')
patch(A, '    ap.add_argument("--outdir", default="analysis_out")\n',
         '    ap.add_argument("--outdir", default="analysis_out")\n'
         '    ap.add_argument("--probe", choices=["auto", "sgd"], default="auto",\n'
         '                    help="sgd: every figure uses the SGDClassifier probe "\n'
         '                         "(lp_curve_sgd.csv), even where lp_curve.csv exists")\n')
patch(A, '    has_lr = any(e["deep"] for e in data.values())\n',
         '    has_lr = any(e["deep"] for e in data.values()) and args.probe != "sgd"\n')
patch(A, '    lp_key, lp_lbl = ("lp", "LP") if any(e["lp"] for e in data.values()) else ("lpsgd", "LP-SGD")\n',
         '    lp_key, lp_lbl = ("lp", "LP") if has_lr and any(e["lp"] for e in data.values()) else ("lpsgd", "LP-SGD")\n')
patch(A, '    plot_curves_by_epoch(data, args.outdir)\n',
         '    plot_curves_by_epoch(data, args.outdir,\n'
         '                         "lpsgd_curve" if args.probe == "sgd" else "lp_curve")\n')
patch(A, '    plot_per_task_by_buffer(records, args.outdir, ref)\n',
         '    if args.probe != "sgd":\n'
         '        plot_per_task_by_buffer(records, args.outdir, ref)\n')

F = "scripts/analysis/forgetting_by_epochs.py"
patch(F, 'def deep_from_lp(run_dir):\n    f = glob.glob(os.path.join(run_dir, "lp_curve*.csv"))\n',
         'PROBE_FILE = None  # set by --probe-file; None = first lp_curve*.csv found\n\n\n'
         'def deep_from_lp(run_dir):\n'
         '    pat = PROBE_FILE or "lp_curve*.csv"\n'
         '    f = sorted(glob.glob(os.path.join(run_dir, pat)))\n')
patch(F, '    roots = sys.argv[1:] or ["."]\n',
         '    global PROBE_FILE\n'
         '    argv = sys.argv[1:]\n'
         '    if "--probe-file" in argv:\n'
         '        i = argv.index("--probe-file")\n'
         '        PROBE_FILE = argv[i + 1]\n'
         '        argv = argv[:i] + argv[i + 2:]\n'
         '    roots = argv or ["."]\n')

N = "scripts/analysis/plot_nc.py"
patch(N, 'EPOCH_STYLE = {1: ("#1f77b4", "o"), 3: ("#d95f02", "s"), 5: ("#2ca02c", "^")}',
         'EPOCH_STYLE = {1: ("#1f77b4", "o"), 3: ("#d95f02", "s"), 5: ("#2ca02c", "^"),\n'
         '               20: ("#7570b3", "D"), 50: ("#e7298a", "v")}')

print("patch OK")
