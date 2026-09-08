#!/usr/bin/env python3
"""Neural Collapse metrics (NC1, NC2, NC3) over training, for the multi-head TIL setup.

Statistics are computed on each task's own data.
"""
import argparse
import csv
import itertools
import os
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from omegaconf import OmegaConf  # noqa: E402

import src.factories.benchmark_factory as benchmark_factory  # noqa: E402
import src.toolkit.utils as utils  # noqa: E402
from src.factories.benchmark_factory import DS_CLASSES  # noqa: E402
from src.toolkit.representation_analysis import (extract_features,  # noqa: E402
                                                 list_checkpoints,
                                                 load_mt_checkpoint,
                                                 task_experience_map)


def rebuild_benchmark(run_dir):
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    utils.set_seed(cfg.experiment.seed)
    fa = OmegaConf.to_container(cfg.benchmark.factory_args, resolve=True)
    benchmark = benchmark_factory.create_benchmark(
        **fa, dataset_root=cfg.benchmark.dataset_root
    )
    n_cls = DS_CLASSES[fa["benchmark_name"]] // fa["n_experiences"]
    return benchmark, n_cls


def class_means_and_within(feats, labels):
    """Per-class means [K,d], global mean [d], and within-class scatter Sigma_W [d,d]."""
    classes = sorted(torch.unique(labels).tolist())
    mus, Sw, n_tot = [], None, 0
    for c in classes:
        fc = feats[labels == c].double()
        mu = fc.mean(0)
        mus.append(mu)
        d = fc - mu
        S = d.T @ d
        Sw = S if Sw is None else Sw + S
        n_tot += fc.shape[0]
    M = torch.stack(mus)                 # [K, d]
    return M, M.mean(0), Sw / n_tot, classes


def nc1(M, mu_G, Sw):
    """Tr(Sigma_W Sigma_B^dagger) / K -- Papyan et al. variability-collapse ratio."""
    Mc = M - mu_G                         # centered class means [K, d]
    K = Mc.shape[0]
    Sb = (Mc.T @ Mc) / K                  # between-class covariance [d, d], rank K-1
    Sb_pinv = torch.linalg.pinv(Sb, hermitian=True)
    return float(torch.trace(Sw @ Sb_pinv) / K)


def nc2(M, mu_G):
    """(equinorm CV, mean pairwise cosine, std of those cosines)."""
    Mc = M - mu_G
    norms = torch.linalg.norm(Mc, dim=1)
    equinorm = float(norms.std(unbiased=False) / norms.mean())
    Mn = Mc / norms.unsqueeze(1).clamp_min(1e-12)
    cos = [float(Mn[a] @ Mn[b]) for a, b in itertools.combinations(range(Mc.shape[0]), 2)]
    return equinorm, float(np.mean(cos)), float(np.std(cos))


def nc3(W, M, mu_G):
    """Frobenius alignment between head weights and centered class means, optimum 1.

    Paper (Prop. 1, NC3): W_h^T/||W_h|| = U~/||U~||. Both sides are normalised to unit
    Frobenius norm, so their Frobenius inner product is 1 exactly when they coincide.
    """
    Mc = (M - mu_G)
    A = W.double()
    A = A / torch.linalg.norm(A)
    B = Mc / torch.linalg.norm(Mc)
    return float((A * B).sum())


def head_weight(model, t):
    """The [K, d] weight matrix of task t's head, restricted to its own K classes.

    This benchmark assigns classes to tasks in sequential global blocks (task t owns
    ids [t*K, ..., (t+1)*K-1], not reset to 0 per task), and avalanche's
    IncrementalClassifier grows out_features to cover the highest id it has seen --
    so task t's raw weight matrix has (t+1)*K rows, of which only K are real.
    """
    head = model.linear.classifiers[str(t)]
    full = None
    for obj in (head, getattr(head, "classifier", None)):
        w = getattr(obj, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() == 2:
            full = w
            break
    if full is None:
        raise RuntimeError(f"cannot locate a 2-D weight for head {t}")

    mask = getattr(model.linear, f"active_units_T{t}", None)
    if mask is None:
        raise RuntimeError(f"no active_units buffer for head {t}")
    mask = mask.to(torch.bool)
    idx = torch.arange(mask.shape[0], device=mask.device)[mask]
    return full[idx.to(full.device)].detach().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--stride", type=int, default=5,
                    help="use every S-th checkpoint (NC is expensive: full covariances)")
    ap.add_argument("--split", choices=["train", "test"], default="train",
                    help="which split the statistics are computed on. The paper uses "
                         "training data; 'test' measures population geometry instead")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--train-per-class", type=int, default=0,
                    help="cap samples per class (0 = all). Covariance estimates degrade "
                         "quickly below a few hundred samples per class")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    benchmark, n_cls = rebuild_benchmark(args.run_dir)
    train_map = task_experience_map(benchmark.train_stream)
    test_map = task_experience_map(benchmark.test_stream)
    data_map = train_map if args.split == "train" else test_map

    all_ckpts = list_checkpoints(args.run_dir)
    checkpoints = all_ckpts[:: args.stride]
    if not checkpoints:
        print("No checkpoints found.")
        return
    if all_ckpts and checkpoints[-1] != all_ckpts[-1]:
        # striding can skip the true final checkpoint (its index need not be a
        # multiple of stride) -- always keep it, or A*_final ends up measured
        # hundreds of steps early (same bug fixed earlier in compute_lp.py)
        checkpoints.append(all_ckpts[-1])
    cap = args.train_per_class * n_cls if args.train_per_class else None

    out = args.out or os.path.join(args.run_dir, "nc_curve.csv")
    done = set()
    if os.path.exists(out):
        with open(out, newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                done.add(int(row[0]))
        if done:
            print(f"Resuming: {len(done)} checkpoints already in {out}")

    write_header = not os.path.exists(out)
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if write_header:
        w.writerow(["step", "task", "nc1", "nc2_equinorm", "nc2_cos_mean",
                    "nc2_cos_std", "nc3", "cross_task_cos"])

    print(f"{len(checkpoints)} checkpoints | split={args.split} | K={n_cls} "
          f"| NC2 optimum = {-1/(n_cls-1):.4f} | device={args.device}")

    for step, path in checkpoints:
        if step in done:
            continue
        model, task_ids = load_mt_checkpoint(path, n_cls, train_map, args.device)
        rows, centered = [], {}

        for t in task_ids:
            feats, labels = extract_features(model, data_map[t].dataset, args.device,
                                             max_samples=cap)
            M, mu_G, Sw, _ = class_means_and_within(feats, labels)
            centered[t] = (M - mu_G)
            v1 = nc1(M, mu_G, Sw)
            eq, cm, cs = nc2(M, mu_G)
            try:
                v3 = nc3(head_weight(model, t), M, mu_G)
            except Exception as e:
                print(f"  [warn] NC3 unavailable for task {t}: {e}")
                traceback.print_exc()
                print(f"  [debug] full.device={head_weight.last_full_device} "
                      f"idx.device={head_weight.last_idx_device}", flush=True)
                v3 = float("nan")
            rows.append([step, t, v1, eq, cm, cs, v3])

        # cross-task NC2: cosines between centered means of *different* tasks
        cross = []
        for a, b in itertools.combinations(sorted(centered), 2):
            A = centered[a] / torch.linalg.norm(centered[a], dim=1, keepdim=True).clamp_min(1e-12)
            B = centered[b] / torch.linalg.norm(centered[b], dim=1, keepdim=True).clamp_min(1e-12)
            cross.append(float((A @ B.T).mean()))
        xt = float(np.mean(cross)) if cross else float("nan")
        for r in rows:
            r.append(xt)

        w.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())
        print(f"step {step:6d} | tasks {task_ids} | "
              f"NC1 {np.mean([r[2] for r in rows]):7.3f} | "
              f"NC2cos {np.mean([r[4] for r in rows]):+.4f} | "
              f"NC3 {np.nanmean([r[6] for r in rows]):.3f} | cross {xt:+.4f}")
    fh.close()
    print(f"\nNC curve written to {out}")


if __name__ == "__main__":
    main()
