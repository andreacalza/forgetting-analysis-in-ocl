#!/usr/bin/env python3
"""Joint-training upper bound: the same architecture, retrained from scratch per prefix.

Here the backbone itself is retrained without the constraint,
so the gap to the online curve is the cost of being online.

Per boundary j the script reports:
    acc      mean per-task Top-1 over tasks 0..j, each task scored through its own head
    snr      mean over tasks of the within-task class-pair SNR, same definition as
             compute_snr.py, so it can be overlaid on the SNR figures
    lp       mean over tasks of a per-task LogisticRegression probe on the frozen
             features of THIS model (the deep counterpart of acc)

Note on batching: this benchmark uses global sequential class ids, so head t has
(t+1)*K outputs and heads have different widths.

"""
import argparse
import csv
import itertools
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from omegaconf import OmegaConf  # noqa: E402

import avalanche.models as models  # noqa: E402
from avalanche.models.dynamic_modules import avalanche_model_adaptation  # noqa: E402

import src.factories.benchmark_factory as benchmark_factory  # noqa: E402
import src.toolkit.utils as utils  # noqa: E402
from src.factories.benchmark_factory import DS_CLASSES  # noqa: E402
from src.toolkit.representation_analysis import (backbone_features,  # noqa: E402
                                                 extract_features,
                                                 task_experience_map)


def rebuild_benchmark(run_dir):
    """Same seed, same factory args -> the identical class split the run used."""
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    utils.set_seed(cfg.experiment.seed)
    fa = OmegaConf.to_container(cfg.benchmark.factory_args, resolve=True)
    benchmark = benchmark_factory.create_benchmark(
        **fa, dataset_root=cfg.benchmark.dataset_root
    )
    n_cls = DS_CLASSES[fa["benchmark_name"]] // fa["n_experiences"]
    return benchmark, cfg, n_cls, fa["n_experiences"]


def fresh_model(n_cls, train_map, tasks, device):
    """A brand-new multi-head model with heads adapted for exactly `tasks`."""
    model = models.MTSlimResNet18(n_cls)
    for t in tasks:
        avalanche_model_adaptation(model, train_map[t])
    return model.to(device)


def train_joint(model, datasets, epochs, batch_size, device, lr, momentum,
                weight_decay, cosine=True, log_every=5):
    """Offline training on the union of the given task datasets.

    """
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
             if cosine else None)
    loader = DataLoader(ConcatDataset(datasets), batch_size=batch_size,
                        shuffle=True, drop_last=True, num_workers=0)
    model.train()
    for ep in range(epochs):
        tot, n, correct = 0.0, 0, 0
        for batch in loader:
            x, y, tl = batch[0].to(device), batch[1].to(device), batch[2]
            opt.zero_grad()
            # one forward per task present in the batch: heads differ in width,
            # and the single-task path is the one that masks inactive units
            loss, seen = 0.0, 0
            for t in torch.unique(tl).tolist():
                m = (tl == t)
                logits = model(x[m], int(t))
                loss = loss + F.cross_entropy(logits, y[m], reduction="sum")
                correct += (logits.argmax(1) == y[m]).sum().item()
                seen += int(m.sum())
            loss = loss / seen
            loss.backward()
            opt.step()
            tot += float(loss) * seen
            n += seen
        if sched is not None:
            sched.step()
        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            print(f"    epoch {ep + 1:3d}/{epochs} | loss {tot / max(n, 1):.4f} "
                  f"| train acc {correct / max(n, 1):.4f} "
                  f"| lr {opt.param_groups[0]['lr']:.4f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def task_accuracy(model, dataset, task, device, batch_size=128):
    """Top-1 on one task, scored through that task's own head (task-aware, as in TIL)."""
    loader = DataLoader(dataset, batch_size=batch_size)
    correct = total = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        pred = model(x, int(task)).argmax(1)
        correct += (pred == y).sum().item()
        total += len(y)
    return correct / max(total, 1)


def task_snr(feats, labels):
    """Mean over within-task class pairs of ||mu1-mu2||^2 / Tr(S1+S2).

    Same definition as compute_snr.py, so the numbers can sit on the same axis.
    """
    classes = sorted(torch.unique(labels).tolist())
    stats = {}
    for c in classes:
        f = feats[labels == c].double()
        # unbiased=True to match compute_snr.py exactly; with ~100 test samples per
        # class the n/(n-1) factor is small but the two curves must be on one axis
        stats[c] = (f.mean(0), f.var(0, unbiased=True).sum())
    vals = []
    for a, b in itertools.combinations(classes, 2):
        mu_a, va = stats[a]
        mu_b, vb = stats[b]
        denom = float(va + vb)
        if denom > 0:
            vals.append(float(((mu_a - mu_b) ** 2).sum()) / denom)
    return float(np.mean(vals)) if vals else float("nan")


def task_probe(model, train_ds, test_ds, device, C, cap):
    Xtr, ytr = extract_features(model, train_ds, device, max_samples=cap or None)
    Xte, yte = extract_features(model, test_ds, device)
    scaler = StandardScaler().fit(Xtr.numpy())
    clf = LogisticRegression(C=C, max_iter=3000)
    clf.fit(scaler.transform(Xtr.numpy()), ytr.numpy())
    return clf.score(scaler.transform(Xte.numpy()), yte.numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="any run of the sweep; used for its benchmark "
                                    "split, seed and optimiser settings")
    ap.add_argument("--epochs", type=int, default=50,
                    help="offline epochs per boundary. The point of the bound is to be "
                         "unconstrained, so this should be enough to converge -- not the "
                         "1/3/5 of the online runs")
    ap.add_argument("--batch-size", type=int, default=128,
                    help="offline batch size. The run's train_mb_size (10) is an online "
                         "streaming choice and would only slow this down")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--no-cosine", action="store_true",
                    help="keep the learning rate constant instead of annealing it")
    ap.add_argument("--use-config-optim", action="store_true",
                    help="use the run's own optimiser settings instead of the offline "
                         "defaults. Those were tuned for online streaming, so expect a "
                         "weaker (i.e. underestimated) bound")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--C", type=float, default=100.0, help="probe C (paper: 100)")
    ap.add_argument("--train-per-class", type=int, default=200,
                    help="cap on samples per class for the probe fit, matching "
                         "compute_upper_bound.py")
    ap.add_argument("--no-lp", action="store_true",
                    help="skip the linear probe (accuracy and SNR only)")
    ap.add_argument("--save-models", action="store_true",
                    help="keep each boundary's trained weights next to the CSV")
    ap.add_argument("--outdir", default="analysis_joint_ub",
                    help="where the result goes. Deliberately NOT the run directory: "
                         "this bound is independent of buffer size and epochs")
    ap.add_argument("--out", default=None, help="explicit CSV path, overrides --outdir")
    args = ap.parse_args()

    benchmark, cfg, n_cls, n_tasks = rebuild_benchmark(args.run_dir)
    train_map = task_experience_map(benchmark.train_stream)
    test_map = task_experience_map(benchmark.test_stream)

    seed = int(cfg.experiment.seed)
    out = args.out or os.path.join(args.outdir, f"joint_upper_bound_seed{seed}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    done = set()
    if os.path.exists(out):
        with open(out, newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                done.add(int(row[0]))
        if done:
            print(f"Resuming: boundaries {sorted(done)} already in {out}")

    write_header = not os.path.exists(out)
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if write_header:
        w.writerow(["boundary_task", "n_tasks", "n_classes", "acc", "snr", "lp",
                    "per_task_acc"])

    if args.use_config_optim:
        lr, momentum, wd = (cfg.optimizer.lr, cfg.optimizer.momentum,
                            cfg.optimizer.weight_decay)
        src = "run config (online)"
    else:
        lr, momentum, wd = args.lr, args.momentum, args.weight_decay
        src = "offline defaults"
    print(f"joint upper bound | {n_tasks} boundaries | {args.epochs} epochs each "
          f"| batch {args.batch_size} | device {args.device}")
    print(f"optimiser [{src}]: SGD lr={lr} momentum={momentum} weight_decay={wd} "
          f"cosine={not args.no_cosine}")
    cap = args.train_per_class * n_cls if args.train_per_class else 0

    for j in range(n_tasks):
        if j in done:
            continue
        tasks = list(range(j + 1))
        print(f"\n=== boundary {j} : training from scratch on tasks {tasks} "
              f"({(j + 1) * n_cls} classes) ===", flush=True)

        model = fresh_model(n_cls, train_map, tasks, args.device)
        train_joint(model, [train_map[t].dataset for t in tasks],
                    args.epochs, args.batch_size, args.device,
                    lr, momentum, wd, cosine=not args.no_cosine)

        accs, snrs, lps = [], [], []
        for t in tasks:
            accs.append(task_accuracy(model, test_map[t].dataset, t, args.device))
            feats, labels = extract_features(model, test_map[t].dataset, args.device)
            snrs.append(task_snr(feats, labels))
            if not args.no_lp:
                lps.append(task_probe(model, train_map[t].dataset,
                                      test_map[t].dataset, args.device, args.C, cap))

        acc, snr = float(np.mean(accs)), float(np.mean(snrs))
        lp = float(np.mean(lps)) if lps else ""
        w.writerow([j, j + 1, (j + 1) * n_cls, acc, snr, lp,
                    " ".join(f"{a:.4f}" for a in accs)])
        fh.flush()
        os.fsync(fh.fileno())
        print(f"    -> acc {acc:.4f} | snr {snr:.4f}"
              + (f" | lp {lp:.4f}" if lps else ""), flush=True)

        if args.save_models:
            torch.save(model.state_dict(),
                       os.path.join(os.path.dirname(os.path.abspath(out)),
                                    f"joint_ub_seed{seed}_model_{j}.ckpt"))
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    fh.close()
    print(f"\nJoint upper bound written to {out}")


if __name__ == "__main__":
    main()
