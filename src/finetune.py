"""
finetune.py — fine-tune an existing corrector checkpoint on an arbitrary set of true .trees
===========================================================================================
train.py discovers its data through the {scenario}_S{size}_{i}.trees naming produced by our
generators; that coupling is deliberate for reproducing the baseline.  This script is the
generic counterpart: point it at ANY directory and a filename SUFFIX and it fine-tunes the
given checkpoint on every matching tree sequence, then reports whether it actually helped.

  - Model + architecture are INFERRED from the checkpoint (model.infer_and_load) — you pass only
    the path, exactly like the diagnostics.  Fine-tuning starts from the loaded weights.
  - Data is any directory of GROUND-TRUTH tree sequences WITH mutations (msprime, SLiM, …): the
    true genealogy is the supervised target, its genotypes are the model input.  The SUFFIX selects
    which files (e.g. "slim.trees"), so inferred trees (_relate/_tsinfer/_singer.trees) are excluded.
  - Simple FILE-level train/test split (--test-frac): the test files are held out entirely; the
    held-out test loss picks the best epoch (as train.py does with its held-out set), and the final
    comparison runs on those same test files.  No separate validation set.
  - At the end it evaluates the BASE (pre-fine-tune) and the FINE-TUNED model on the held-out test
    files and prints fine-tuned-vs-base on cophenetic corr (↑), RF-to-truth (↓), and triplet
    accuracy (↑), broken down PER leaf count S (plus a pooled row when several sizes are present),
    with the input tree (local UPGMA, as in train.py's end table) as a reference column and one row
    per held-out file.  Every held-out file gives at most GNN_FT_EVAL_MAXPOS positions, spread over
    the whole file.
  - The node-feature window (GNN_HAP_WINDOW) and input-graph knobs (GNN_KNN_K, GNN_SEQ_LEN, …) must
    match how the base model was trained; the feature width is checked against the checkpoint and
    errors on mismatch.  Loss = the same cophenetic objective as train.py (corr + mse + rank).
  - The log has the lines of train.py's log: the windows and held-out files per size, the loading
    time, the parameters, and under each epoch line (epoch 0 included) the training and held-out
    loss per size with its parts, then the time of the epoch and the learning rate.

Run with:
  conda run --no-capture-output -n lplancheenv python -u finetune.py \
      --ckpt gnn_v7_varsize_rk1.0L3_knn7w_best.pt --data-dir my_sims --suffix slim.trees \
      --out gnn_finetuned.pt --epochs 15 --lr 2e-4
  # match the base model's input config via env if it was non-default, e.g.:
  #   GNN_HAP_WINDOW=… GNN_KNN_K=7 GNN_SEQ_LEN=… conda run … python -u finetune.py …
"""

import argparse
import glob
import math
import os
import random
import time

import numpy as np
import torch
import tskit
from torch.utils.data import DataLoader

import gnn_topology as T
from gnn_topology import (
    cophenetic_corr, cophenetic_corr_loss, cophenetic_mse, cophenetic_rank_loss,
    cophenetic_from_parent, decode_tree_from_mrca, parent_index_to_newick, rf_distance,
    GAMMA, RANK_WEIGHT, RANK_TAU, RANK_LEVELS, RANK_DECAY,
)
from model import infer_and_load, build_parent_index_from_topology, save_flushed
# Same input-graph construction, collate, full-context inference, and triplet metric as the trainer.
from train import (
    build_dataset, collate, _flat_stream, _sliding_predict, decode_model, EVAL_STRIDE,
    _triple_indices, _triplet_labels, wrong_triplet_fraction,
)

torch.manual_seed(42); random.seed(42); np.random.seed(42)

# Scale anchor on the correlation loss — read here (it lives in train.py) so fine-tuning uses the
# same objective as baseline training without importing train.py's dataset-specific config.
MSE_WEIGHT = float(os.environ.get("GNN_MSE_WEIGHT", "1.0"))
EVAL_MAXPOS = int(os.environ.get("GNN_FT_EVAL_MAXPOS", "1000"))   # cap scored positions per held-out file


def _loss_and_parts(pred, true_M):
    """The train.py cophenetic objective (non-cascade) and its unweighted parts [1−corr, MSE, rank], added in
    the order of train.py, so the loss is the same bit for bit."""
    parts = [cophenetic_corr_loss(pred, true_M, gamma=GAMMA), cophenetic_mse(pred, true_M, gamma=GAMMA), 0.0]
    loss = parts[0] + MSE_WEIGHT * parts[1]
    if RANK_WEIGHT > 0:
        parts[2] = cophenetic_rank_loss(pred, true_M, tau=RANK_TAU, levels=RANK_LEVELS, level_decay=RANK_DECAY)
        loss = loss + RANK_WEIGHT * parts[2]
    return loss, parts


def _loss(pred, true_M):
    """The train.py cophenetic objective (non-cascade): (1−corr) + MSE_WEIGHT·MSE + rank."""
    return _loss_and_parts(pred, true_M)[0]


def add_parts(by_size, S, loss, parts):
    """Add one batch's loss and its unweighted parts (1 - Pearson, MSE, rank) to the sums of size S (as train.py)."""
    acc = by_size.setdefault(S, [0.0, 0.0, 0.0, 0.0, 0])
    for k, v in enumerate([loss] + parts):
        acc[k] += v.item() if torch.is_tensor(v) else float(v)
    acc[4] += 1


def parts_text(acc):
    """Mean loss and mean parts of one size, from the sums of add_parts (as train.py)."""
    return (f"{acc[0] / acc[4]:.4f} (1-pearson {acc[1] / acc[4]:.4f}, mse {acc[2] / acc[4]:.4f}, "
            f"rank {acc[3] / acc[4]:.4f})")


def print_sizes(tr_by_size, va_by_size):
    """One line per leaf count under an epoch line: training and held-out (VA) loss with its parts (as train.py)."""
    for S in sorted(tr_by_size):
        va = parts_text(va_by_size[S]) if S in va_by_size else "n/a"
        print(f"      S={S:<4} TR {parts_text(tr_by_size[S])} | VA {va}", flush=True)


def discover(data_dir, suffix):
    """Group ground-truth tree sequences under `data_dir` whose filename ENDS WITH `suffix`,
    keyed by leaf count S (= ts.num_samples).  Returns {S: [(path, ts), ...]}."""
    paths = sorted(p for p in glob.glob(os.path.join(data_dir, "*")) if p.endswith(suffix))
    by_size = {}
    for p in paths:
        ts = tskit.load(p)
        by_size.setdefault(ts.num_samples, []).append((p, ts))
    return by_size


def split_files(by_size, test_frac):
    """FILE-level train/test split within each leaf count (so batches stay homogeneous-size and the
    test set is leakage-free).  Returns (train_by_size, test_by_size) as {S: [(path, ts), ...]}."""
    train_by_size, test_by_size = {}, {}
    for S in sorted(by_size):
        files = by_size[S][:]
        random.shuffle(files)
        n = len(files)
        n_test = max(1, round(test_frac * n)) if n >= 2 else 0     # keep ≥1 test file when possible
        if files[n_test:]:
            train_by_size[S] = files[n_test:]
        if files[:n_test]:
            test_by_size[S] = files[:n_test]
    return train_by_size, test_by_size


def make_loaders(files_by_size, batch, shuffle):
    """One DataLoader per leaf count (homogeneous-size batches); the training loop interleaves them."""
    loaders = {}
    for S, files in files_by_size.items():
        seqs = []
        for _, ts in files:
            seqs += T.createSequenceFromTrees(ts, S)
        if seqs:
            loaders[S] = DataLoader(build_dataset(seqs, S), batch_size=batch, shuffle=shuffle, collate_fn=collate)
    return loaders


def eval_loss(model, loaders, device):
    """Mean cophenetic loss + corr over a set of loaders (cheap per-epoch held-out monitor), and per leaf count
    the sums of the loss and its parts (add_parts)."""
    model.eval()
    tot, ncorr, n = 0.0, 0.0, 0
    by_size = {}
    with torch.no_grad():
        for S, ld in loaders.items():
            for steps, true_M, _ in ld:
                steps = [b.to(device) for b in steps]; true_M = true_M.to(device)
                pred = model(steps)["mrca"]
                loss, parts = _loss_and_parts(pred, true_M)
                tot += loss.item(); ncorr += cophenetic_corr(pred, true_M); n += 1
                add_parts(by_size, S, loss, parts)
    return ((tot / n, ncorr / n) if n else (float("nan"), float("nan"))) + (by_size,)


def eval_metrics(model, test_by_size, device):
    """Full-context (sliding-window) topology scoring on the held-out test files, PER leaf count S and PER file:
    cophenetic corr (↑), RF-to-truth (↓), and triplet accuracy (↑ = 1 − wrong-triplet fraction), of the model and
    of its input tree (the local UPGMA of each window, the "In" columns of train.py's end table).
    Each file gives at most EVAL_MAXPOS positions, every EVAL_STRIDE-th SNP or sparser, spread over the whole file.
    Returns {S: [(file name, {"corr": [...], "rf": [...], "trip": [...], "corr_in": [...], "rf_in": [...],
    "trip_in": [...]}), ...]}."""
    model.eval()
    out = {}
    with torch.no_grad():
        for S in sorted(test_by_size):
            N = 2 * S - 1
            trip = _triple_indices(S)                          # leaf-triple indices, once per size
            out[S] = []
            for path, ts in test_by_size[S]:
                acc = {k: [] for k in ("corr", "rf", "trip", "corr_in", "rf_in", "trip_in")}
                out[S].append((os.path.basename(path), acc))
                items = _flat_stream(ts, S)
                if not items:
                    continue
                preds, in_eis = _sliding_predict(model, items, S, device)
                stride = max(EVAL_STRIDE, math.ceil(len(items) / EVAL_MAXPOS))
                for gp in range(0, len(items), stride):
                    xg, tM, _ = items[gp]
                    ntr = parent_index_to_newick(decode_tree_from_mrca(tM, num_samples=S, num_nodes=N))
                    if ntr is None:
                        continue
                    mod_par = decode_model(preds[gp], S, alleles=xg[:S, T.HAP_K])
                    nmd = parent_index_to_newick(mod_par)
                    pin = build_parent_index_from_topology(in_eis[gp], num_nodes=N, root_idx=N - 1)
                    nin = parent_index_to_newick(pin)
                    in_coph = cophenetic_from_parent(pin, S)        # input-tree cophenetic
                    acc["corr"].append(cophenetic_corr(preds[gp][None, None], tM[None, None]))
                    acc["corr_in"].append(cophenetic_corr(in_coph[None, None], tM[None, None]))
                    for key, nwk in (("rf", nmd), ("rf_in", nin)):
                        if nwk:
                            r = rf_distance(nwk, ntr)
                            if r is not None:
                                acc[key].append(r)
                    if trip is not None:
                        mod_coph = cophenetic_from_parent(mod_par, S)   # decoded-tree cophenetic
                        tlab = _triplet_labels(tM, *trip)
                        acc["trip"].append(1.0 - wrong_triplet_fraction(tlab, mod_coph, *trip))
                        acc["trip_in"].append(1.0 - wrong_triplet_fraction(tlab, in_coph, *trip))
    return out


def _print_comparison(base, ft):
    """Per-size (and, if >1 size, pooled) base-vs-fine-tuned table on the held-out test set, with the input tree
    as a reference column, and under each size one row per held-out file."""
    _m = lambda x: float(np.mean(x)) if x else float("nan")
    keys = ("corr", "rf", "trip", "corr_in", "rf_in", "trip_in")
    merge = lambda files: {k: [v for _, acc in files for v in acc[k]] for k in keys}   # every file of one size
    print(f"\n── Held-out test: fine-tuned vs base (corr & triplet-acc ↑ better, RF ↓ better) ──")

    def block(label, b, f):
        print(f"  {label} ({len(b['corr'])} trees)")
        print(f"    {'metric':<15} {'base':>9} {'fine-tuned':>11} {'Δ (ft−base)':>13} {'input tree':>11}")
        for name, key in (("corr ↑", "corr"), ("RF ↓", "rf"), ("triplet acc ↑", "trip")):
            bb, ff, ii = _m(b[key]), _m(f[key]), _m(b[key + "_in"])
            print(f"    {name:<15} {bb:>9.4f} {ff:>11.4f} {ff - bb:>+13.4f} {ii:>11.4f}")

    for S in sorted(base):
        block(f"S={S}", merge(base[S]), merge(ft[S]))
        print(f"    {'held-out file':<28} {'trees':>6} | {'corr base':>9} {'ft':>7} | {'RF base':>9} {'ft':>9} | "
              f"{'trip base':>9} {'ft':>7}")
        for (name, b), (_, f) in zip(base[S], ft[S]):
            print(f"    {name:<28} {len(b['corr']):>6} | {_m(b['corr']):>9.4f} {_m(f['corr']):>7.4f} | "
                  f"{_m(b['rf']):>9.4f} {_m(f['rf']):>9.4f} | {_m(b['trip']):>9.4f} {_m(f['trip']):>7.4f}")
    if len(base) > 1:                                          # pooled across sizes
        pool = lambda d: {k: [v for S in d for v in merge(d[S])[k]] for k in keys}
        block("ALL sizes pooled", pool(base), pool(ft))


def main():
    ap = argparse.ArgumentParser(description="Fine-tune a corrector checkpoint on a directory of true .trees.")
    ap.add_argument("--ckpt", required=True, help="base checkpoint to fine-tune (architecture is inferred from it)")
    ap.add_argument("--data-dir", required=True, help="directory of ground-truth tree sequences (with mutations)")
    ap.add_argument("--suffix", default=".trees", help="only files whose name ends with this are used (e.g. slim.trees)")
    ap.add_argument("--out", default=None, help="output checkpoint path (default: <ckpt>_ft.pt); must differ from --ckpt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-4, help="fine-tuning LR (lower than the 1e-3 baseline)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--test-frac", type=float, default=0.2, help="fraction of FILES held out for testing")
    args = ap.parse_args()

    out = args.out or (os.path.splitext(args.ckpt)[0] + "_ft.pt")
    if os.path.abspath(out) == os.path.abspath(args.ckpt):
        raise SystemExit("--out must differ from --ckpt (refusing to overwrite the base model)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tune | base={args.ckpt} | data={args.data_dir} suffix='{args.suffix}' | out={out}")
    print(f"          | epochs={args.epochs} lr={args.lr} batch={args.batch} test_frac={args.test_frac} | device={device}")
    print(f"          | input config: SEQ_LEN={T.SEQ_LEN} hap_window={T.HAP_W}(k={T.HAP_K}) "
          f"mse_w={MSE_WEIGHT} rank_w={RANK_WEIGHT}(levels={RANK_LEVELS})")

    model, cfg = infer_and_load(args.ckpt, map_location=device, return_cfg=True)
    model = model.to(device)
    # As in train.py: infer_and_load builds the model without gradient checkpointing.
    model.grad_checkpoint = os.environ.get("GNN_GRAD_CKPT", "0") == "1"
    if cfg["node_feat_dim"] != T.HAP_W:
        raise SystemExit(
            f"checkpoint node-feature width {cfg['node_feat_dim']} != current HAP_W={T.HAP_W}; "
            f"set GNN_HAP_WINDOW={(cfg['node_feat_dim'] - 1) // 2} to match the base model")
    max_S = (cfg["num_nodes"] + 1) // 2

    load_start = time.time()
    by_size = discover(args.data_dir, args.suffix)
    if not by_size:
        raise SystemExit(f"no files ending with '{args.suffix}' found in {args.data_dir}")
    too_big = [S for S in by_size if S > max_S]
    if too_big:
        raise SystemExit(f"leaf counts {sorted(too_big)} exceed the checkpoint's max S={max_S} "
                         f"(num_nodes={cfg['num_nodes']}); fine-tune on S ≤ {max_S} or retrain a larger model")
    n_files = sum(len(v) for v in by_size.values())

    train_by_size, test_by_size = split_files(by_size, args.test_frac)
    train_loaders = make_loaders(train_by_size, args.batch, shuffle=True)
    test_loaders  = make_loaders(test_by_size, args.batch, shuffle=False)
    if not train_loaders:
        raise SystemExit("no trainable windows found (files too short for one SEQ_LEN window?)")
    n_train_files = sum(len(v) for v in train_by_size.values())
    n_test_files  = sum(len(v) for v in test_by_size.values())
    print(f"{n_files} files (sizes {sorted(by_size)}) → {n_train_files} train / {n_test_files} test", flush=True)
    if not test_by_size:
        print("  [note] <2 files for every size → no held-out test set; comparison will be skipped.", flush=True)
    # As train.py: windows per size, the held-out (validation) files, the loading time, and the parameters.
    for S in sorted(by_size):
        n_tr = len(train_loaders[S].dataset) if S in train_loaders else 0
        n_va = len(test_loaders[S].dataset) if S in test_loaders else 0
        names = [os.path.basename(p) for p, _ in test_by_size.get(S, [])]
        print(f"  S={S}: {n_tr} train / {n_va} val seqs, val files {names}", flush=True)
    print(f"  Loading time: {time.time() - load_start:.1f} s", flush=True)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Baseline reference: score the (pre-fine-tune) model on the held-out test set now.
    table_start = time.time()
    base_metrics = eval_metrics(model, test_by_size, device) if test_by_size else None
    print(f"  Base model evaluation time: {time.time() - table_start:.1f} s", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    monitor = test_loaders or train_loaders           # if no test set, track train loss for the ✓
    print(f"\n{'Ep':>3} | {'TR loss':>8} {'TR corr':>7} | {'TE loss':>8} {'TE corr':>7}")
    print("─" * 46)
    print(f"      per size: loss (1-pearson, mse, rank), loss = 1-pearson + {MSE_WEIGHT:g} * mse + "
          f"{RANK_WEIGHT:g} * rank", flush=True)
    # Epoch 0 = the INPUT model, BEFORE any fine-tuning: a clean eval of the loaded weights on the
    # train and monitor sets, so the table shows the starting point every later epoch is measured against.
    tr0_loss, tr0_corr, tr0_by_size = eval_loss(model, train_loaders, device)
    te0_loss, te0_corr, te0_by_size = eval_loss(model, monitor, device)
    print(f"{0:>3} | {tr0_loss:>8.4f} {tr0_corr:>7.4f} | {te0_loss:>8.4f} {te0_corr:>7.4f}  (input model)", flush=True)
    print_sizes(tr0_by_size, te0_by_size)
    best = float("inf")
    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        # As train.py: interleave the per-size loaders in random order, one collated batch at a time.  A list of
        # every collated batch of the epoch took more host memory than the training windows themselves.
        iters = {S: iter(ld) for S, ld in train_loaders.items()}
        schedule = [S for S, ld in train_loaders.items() for _ in range(len(ld))]
        random.shuffle(schedule)
        tot, ncorr, n = 0.0, 0.0, 0
        by_size, batch_time, lr, epoch_start = {}, 0.0, opt.param_groups[0]["lr"], time.time()
        for S in schedule:
            batch_start = time.time()
            steps, true_M, _ = next(iters[S])
            batch_time += time.time() - batch_start
            steps = [b.to(device) for b in steps]; true_M = true_M.to(device)
            pred = model(steps)["mrca"]
            loss, parts = _loss_and_parts(pred, true_M)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
            tot += loss.item(); ncorr += cophenetic_corr(pred.detach(), true_M); n += 1
            add_parts(by_size, S, loss, parts)
        train_time = time.time() - epoch_start
        sched.step()
        te_loss, te_corr, te_by_size = eval_loss(model, monitor, device)
        flag = ""
        if te_loss < best:
            best = te_loss; save_flushed(model, out, verbose=False); flag = " ✓"
        print(f"{epoch:>3} | {tot/n:>8.4f} {ncorr/n:>7.4f} | {te_loss:>8.4f} {te_corr:>7.4f}{flag}", flush=True)
        print_sizes(by_size, te_by_size)
        epoch_time = time.time() - epoch_start
        print(f"      time {epoch_time:.1f} s: training {train_time:.1f} s (building batches {batch_time:.1f} s), "
              f"validation and saving {epoch_time - train_time:.1f} s | lr {lr:.4e}", flush=True)

    total_train_time = time.time() - train_start
    print(f"\nTotal training time: {total_train_time:.1f} s "
          f"({total_train_time / 60:.1f} min) over {args.epochs} epochs", flush=True)
    print(f"\nBest (lowest held-out loss) fine-tuned model → {out}")

    # Final comparison: reload the BEST fine-tuned weights and score them against the base on test.
    if test_by_size:
        table_start = time.time()
        model.load_state_dict(torch.load(out, map_location=device))
        ft_metrics = eval_metrics(model, test_by_size, device)
        _print_comparison(base_metrics, ft_metrics)
        print(f"\nEnd table time: {time.time() - table_start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
