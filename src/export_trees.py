"""
export_trees.py — run the model over a sequence and write a tskit .trees output (topology + TIMES)
==================================================================================================
Takes an input tree sequence (we only use its genotypes + SNP positions to drive the
model — Relate-free, local-greedy input), runs the model with sliding/full-context
inference, decodes a tree at every SNP, and assembles a valid tskit.TreeSequence spanning
the whole sequence.

By DEFAULT the output now carries REAL node TIMES, not just topology: the decode is assembled
into a datable tree sequence (mutations placed by parsimony, node identity shared across positions
by GNN_BL_ASSEMBLY) and dated with GNN_DATING_METHOD (tsdate|polegon) — the exact branch_length.py
pipeline.  So internal-node times are estimated coalescent times.  (Set GNN_EXPORT_DATE=0 for the
old TOPOLOGY-ONLY output whose internal-node times are the SYNTHETIC placeholder "clade size".)

The dated path forces the avg_constr decoder (needed for clean per-clade mutation placement) and
honours the branch_length knobs: GNN_BL_ASSEMBLY (bridge_noroot recommended), GNN_BL_ROOT_METRIC/
_TOL/_CMP (e.g. clade/1/anchor), GNN_DATING_METHOD, GNN_MUT_RATE (must match the data!), GNN_DATING_NE.

Run (match the checkpoint's env, e.g.):
  GNN_CKPT=.../gnn_v7_varsize_best.pt GNN_SEQ_LEN=256 GNN_HIDDEN=128 GNN_GRU_HIDDEN=128 \
  GNN_GRU_LAYERS=2 GNN_MP_LAYERS=4 GNN_METRIC_DIM=64 \
  GNN_BL_ASSEMBLY=bridge_noroot GNN_BL_ROOT_METRIC=clade GNN_BL_ROOT_TOL=1 GNN_BL_ROOT_CMP=anchor \
  conda run --no-capture-output -n lplancheenv python -u export_trees.py INPUT_DIR OUTPUT_DIR
  # every *.trees in INPUT_DIR is exported to OUTPUT_DIR/<same-name>_<GNN_TAG_OUT>.trees

As a function:
  ts = model_to_tree_sequence(model, input_ts, S, device)            # dated (real node times)
  ts = model_to_tree_sequence(model, input_ts, S, device, date=False) # topology only (placeholder times)
"""

import glob
import os
import sys

import numpy as np
import torch
import tskit

import train as V


# ── tree helpers ────────────────────────────────────────────────────────────────
def _clades(parent, S):
    """Set of frozenset(leaves) for the non-trivial clades — used to test topology equality."""
    parent = [int(p) for p in (parent.tolist() if torch.is_tensor(parent) else parent)]
    N = len(parent)
    children = [[] for _ in range(N)]
    root = -1
    for i, p in enumerate(parent):
        children[p].append(i) if p >= 0 else None
        if p < 0:
            root = i
    out = set()

    def rec(n):
        if not children[n]:
            return frozenset([n]) if n < S else frozenset()
        acc = set()
        for c in children[n]:
            acc |= rec(c)
        fs = frozenset(acc)
        if 2 <= len(fs) <= S - 1:
            out.add(fs)
        return fs

    if root >= 0:
        rec(root)
    return frozenset(out)


def _sizes_postorder(parent, S):
    """Per-node clade size + a children list + the root (post-order not needed: sizes suffice)."""
    N = len(parent)
    children = [[] for _ in range(N)]
    root = -1
    for i, p in enumerate(parent):
        p = int(p)
        children[p].append(i) if p >= 0 else None
        if p < 0:
            root = i
    size = [0] * N

    def _sz(n):
        if not children[n]:
            size[n] = 1
        else:
            size[n] = sum(_sz(c) for c in children[n])
        return size[n]

    if root >= 0:
        _sz(root)
    return size


def model_to_tree_sequence(model, input_ts, S, device, date=True):
    """Run the model over `input_ts` and return a tskit.TreeSequence of the predicted trees.

    date=True  (default): mutations placed + node times ESTIMATED — routes through branch_length.py's
        assemble_decoded (avg_constr decode, GNN_BL_ASSEMBLY node-sharing, parsimony mutation placement)
        and date_ts (GNN_DATING_METHOD, GNN_MUT_RATE) ⇒ REAL coalescent times on the nodes.
    date=False: TOPOLOGY ONLY — internal-node times are the synthetic placeholder "clade size"."""
    items = V._flat_stream(input_ts, S)                       # (x, M, pos) per SNP, in order
    if not items:
        raise ValueError("no SNPs in the input sequence")
    with torch.no_grad():
        preds, _ = V._sliding_predict(model, items, S, device)    # full-context matrix per SNP

    if not date:
        return _topology_only_ts(preds, items, float(input_ts.sequence_length), S)

    import branch_length as BL                                # lazy: pulls in tsdate/model only when dating
    V.DECODER = "avg_constr"                                  # clean per-clade mutation placement for dating
    ts, _diag, _ = BL.assemble_decoded(preds, items, S, float(input_ts.sequence_length),
                                       assembly=BL.BL_ASSEMBLY)
    return BL.date_ts(ts, method=BL.DATING_METHOD)


def _topology_only_ts(preds, items, L, S):
    """Legacy assembly: merge consecutive SNPs with identical topology; internal-node times = clade
    size (synthetic placeholders, valid parent>child ordering).  No mutations, no real times."""
    N = 2 * S - 1
    positions = [float(it[2]) for it in items]
    parents = [V.decode_model(preds[k], S).cpu().numpy().astype(int) for k in range(len(items))]
    keys = [_clades(p, S) for p in parents]

    # Per-SNP interval boundaries (midpoints between adjacent SNPs); tile [0, L).
    n = len(positions)
    bounds = [0.0] + [(positions[i] + positions[i + 1]) / 2 for i in range(n - 1)] + [L]

    # Merge consecutive SNPs with identical topology into one interval.
    groups = []                                               # (left, right, parent_array)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and keys[j + 1] == keys[i]:
            j += 1
        groups.append((bounds[i], bounds[j + 1], parents[i]))
        i = j + 1

    # Build tskit tables: leaves once (time 0), fresh internal nodes per interval.
    tables = tskit.TableCollection(sequence_length=L)
    for _ in range(S):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
    for left, right, par in groups:
        size = _sizes_postorder(par, S)
        loc2glob = {c: c for c in range(S)}                  # leaves are global 0..S-1
        for nd in range(S, N):                               # internal nodes → fresh global ids
            loc2glob[nd] = tables.nodes.add_row(flags=0, time=float(size[nd]))
        for child in range(N):
            p = int(par[child])
            if p >= 0:
                tables.edges.add_row(left=left, right=right,
                                     parent=loc2glob[p], child=loc2glob[child])
    tables.sort()
    return tables.tree_sequence()


def _build_model_from_ckpt(sd, device):
    """Reconstruct the model from checkpoint shapes (same inference as build_cascade)."""
    num_nodes   = sd["node_types"].shape[0] if "node_types" in sd else sd["parent_mask"].shape[0]
    S           = (num_nodes + 1) // 2
    type_dim    = sd["type_embed.weight"].shape[1] if "type_embed.weight" in sd else 0
    node_feat   = sd["node_input_proj.weight"].shape[1] - type_dim
    node_hidden = sd["node_input_proj.weight"].shape[0]
    gru_hidden  = sd["node_gru.weight_ih_l0"].shape[0] // 3
    n_gru       = len([k for k in sd if k.startswith("node_gru.weight_ih_l") and "reverse" not in k])
    mp          = len([k for k in sd if k.startswith("mp_layers.") and k.endswith(".message_mlp.0.weight")])
    metric      = sd["metric_proj.2.weight"].shape[0]
    jk          = ("cat" if "jk_proj.weight" in sd else "max" if "jk_max" in sd else "sum" if "jk_sum" in sd
                   else (os.environ.get("GNN_JK", "").lower() or None))
    m = V.GraphSequenceCorrector(
        node_feat_dim=node_feat, edge_feat_dim=1, num_nodes=num_nodes, num_samples=S,
        node_hidden=node_hidden, num_mp_layers=mp, gru_hidden=gru_hidden, num_gru_layers=n_gru,
        parent_hidden=32, node_id_dim=0, type_dim=type_dim, topology_only=True,
        pairwise_mrca=True, metric_dim=metric, jk_mode=jk).to(device)
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m, node_feat


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python export_trees.py INPUT_DIR OUTPUT_DIR")
    in_dir, out_dir = sys.argv[1], sys.argv[2]
    ckpt = os.environ.get("GNN_CKPT", V.SAVE_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag_out = os.environ.get("GNN_TAG_OUT", "none")
    print(f"device {device} | ckpt {ckpt} | decoder {V.DECODER}", flush=True)

    model, node_feat = _build_model_from_ckpt(torch.load(ckpt, map_location=device), device)
    if node_feat != V.HAP_W:
        raise SystemExit(f"checkpoint hap window W={node_feat} != GNN_HAP_WINDOW ({V.HAP_W}); set it to match")

    in_paths = sorted(glob.glob(os.path.join(in_dir, "*.trees")))     # every top-level .trees in the input dir
    if not in_paths:
        raise SystemExit(f"no .trees files in {in_dir}")
    os.makedirs(out_dir, exist_ok=True)
    date = os.environ.get("GNN_EXPORT_DATE", "1") not in ("0", "false", "False", "")

    for in_path in in_paths:
        input_ts = tskit.load(in_path)
        S = input_ts.num_samples
        print(f"input {in_path}: S={S} samples, {input_ts.num_sites} sites, "
            f"{input_ts.sequence_length/1e6:.1f} Mb", flush=True)

        ts = model_to_tree_sequence(model, input_ts, S, device, date=date)
        base = os.path.splitext(os.path.basename(in_path))[0]        # same name + the tag
        out_file = os.path.join(out_dir, f"{base}_{tag_out}.trees")
        ts.dump(out_file)
        if date:
            import branch_length as BL
            print(f"wrote {out_file}: {ts.num_trees} trees, {ts.num_nodes} nodes, {ts.num_mutations} mutations "
                f"(node times = {BL.DATING_METHOD}; assembly={BL.BL_ASSEMBLY}; mut_rate={BL.MUT_RATE})", flush=True)
        else:
            print(f"wrote {out_file}: {ts.num_trees} trees, {ts.num_nodes} nodes "
              f"(synthetic internal-node times = clade size; topology only)", flush=True)


if __name__ == "__main__":
    main()
