"""
train.py — one model trained across multiple LEAF COUNTS (variable-size trainer)
===============================================================================
Trains the size-dynamic, type-embedding pairwise model on a mix of leaf counts and
evaluates per-size, including a HELD-OUT size to test generalization.

The size-agnostic core (config, losses/metrics, decoders, and the S-parametric
topology helpers) lives in gnn_topology.py; this file holds the variable-size
training/eval pipeline (kNN input edges, cascade scaffolds, triplet metrics,
sliding-window inference).

Key ingredients that make variable size work:
  - model is size-dynamic (infers N, S from each batch) + type embedding (size-agnostic).
  - target = clade-size cophenetic NORMALISED by S (∈(0,1]) so MSE is comparable across sizes;
    the correlation loss is already scale/size-invariant.
  - homogeneous-size batches: one DataLoader per size, batches interleaved each epoch
    (the model handles each batch's S; loss/decode infer S from shape).
  - local-greedy input (Relate-free), derived from each window's genotypes.

Run with:
  conda run --no-capture-output -n lplancheenv python -u train.py
Env: GNN_VS_TRAIN_SIZES ("10,20,50"), GNN_VS_TEST_SIZES ("100"),
     GNN_HIDDEN/GNN_GRU_HIDDEN/GNN_GRU_LAYERS/GNN_MP_LAYERS/GNN_METRIC_DIM/GNN_TYPE_DIM,
     GNN_DECODER (avg_constr), GNN_GAMMA (0), GNN_MSE_WEIGHT (1.0), GNN_EPOCHS (40).
"""

import bisect
import os
import random
import time

import numpy as np
import torch
import tskit
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from gnn_topology import (
    SEQ_LEN, NODE_FEAT_DIM, EDGE_FEAT_DIM, TYPE_DIM, METRIC_DIM,
    NODE_HIDDEN, GRU_HIDDEN, GRU_LAYERS, MP_LAYERS, GAMMA,
    HAP_K, HAP_W, encode_genotype_matrix, hap_window,
    tskit_tree_to_pyg, tree_cophenetic, createSequenceFromTrees,
    local_upgma_edges, cophenetic_from_parent,
    RANK_WEIGHT, RANK_TAU, RANK_LEVELS, RANK_DECAY, cophenetic_rank_loss, cophenetic_anchor_loss, JK_MODE,
    cophenetic_corr_loss, cophenetic_mse, cophenetic_corr,
    decode_tree_from_mrca, decode_tree_nj, decode_tree_constrained, decode_tree_nj_constrained,
    parent_index_to_newick, rf_distance,
)
from model import (
    GraphSequenceCorrector, build_parent_index_from_topology, save_flushed,
)

torch.manual_seed(42); random.seed(42); np.random.seed(42)

DATA_DIR    = os.environ.get("GNN_VS_DIR",
                             os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulated_varsize"))
TRAIN_SIZES = [int(x) for x in os.environ.get("GNN_VS_TRAIN_SIZES", "10,20,50").split(",") if x.strip()]
TEST_SIZES  = [int(x) for x in os.environ.get("GNN_VS_TEST_SIZES", "100").split(",") if x.strip()]
# Demographic scenarios to pool over (files {scenario}_S{size}_{i}.trees). One model is
# trained across ALL of them; the final eval reports each scenario separately + combined.
# Scenarios with no files present are silently skipped (so const-only data still works).
SCENARIOS   = [s for s in os.environ.get("GNN_VS_SCENARIOS", "const,split,bott").split(",") if s]
FILES_PER   = int(os.environ.get("GNN_VS_FILES", "24"))
BATCH_SIZE  = int(os.environ.get("GNN_BATCH", "16"))
EPOCHS      = int(os.environ.get("GNN_EPOCHS", "30"))
LR          = float(os.environ.get("GNN_LR", "1e-3"))
MSE_WEIGHT  = float(os.environ.get("GNN_MSE_WEIGHT", "1.0"))   # higher: target is normalised (small)
# Decoder for the model's predicted matrix at eval: "avg" (UPGMA) | "nj" (neighbour-joining)
# | "avg_constr" (UPGMA forcing the focal SNP's bipartition to be an edge).  Module-global so
# diagnostic scripts can flip it at runtime (e.g. `import train as V; V.DECODER = "nj"`).
DECODER     = os.environ.get("GNN_DECODER", "avg_constr").lower()
# Input topology for stage-2: "local_greedy" (default, UPGMA per window) | "cascade"
# (a frozen stage-1 model's decoded tree per position, precomputed by build_cascade_dataset.py
# into const_S{S}_{i}_stage1_{STAGE1_TAG}.npz).  Cascade gives stage-2 a corrected DEEP scaffold.
VS_INPUT    = os.environ.get("GNN_VS_INPUT", "local_greedy").lower()
STAGE1_TAG  = os.environ.get("GNN_STAGE1_TAG", "")
# Deep-anchor (cascade): a training regulariser pulling the prediction toward stage-1's
# CONTINUOUS matrix, weighted toward DEEP pairs — preserves the scaffold's deep structure
# while the rank loss re-resolves cherries.  The model predicts normally (no baseline at
# inference), so the backbone gradient stays healthy.  Needs cascade input + coph in the cache.
ANCHOR_WEIGHT = float(os.environ.get("GNN_ANCHOR_WEIGHT", "0.0")) if VS_INPUT == "cascade" else 0.0
# Genotype-kNN shortcut edges: augment the local-UPGMA input graph with an undirected edge
# from each leaf to its k genotype-nearest leaves (hamming on the window's focal SNPs). True
# neighbours the UPGMA tree split far apart can then exchange messages within mp hops, instead
# of being unreachable — directly attacks the misplaced-cherry / under-reaching limit.
# The (1-dim) edge feature carries TYPE via sign and STRENGTH via magnitude (EDGE_FEAT_DIM stays
# 1, no arch change): tree/scaffold = +TREE_EDGE_VAL; kNN = −similarity (weighted, default) or
# KNN_EDGE_VAL (flat).  0 = off (byte-identical legacy).
KNN_K       = int(os.environ.get("GNN_KNN_K", "0"))
# Similarity-WEIGHTED kNN edges (default): kNN edge feature = −(genotype similarity ∈[0,1]) so the
# MPNN attenuates weak/spurious edges (loners, no-cherry-signal leaves) that otherwise force false
# groups and over-balance the tree.  Adapts to where real cherry signal exists; size-robust.
# GNN_KNN_WEIGHTED=0 → flat −1 edges (old behaviour, for A/B).
KNN_WEIGHTED = os.environ.get("GNN_KNN_WEIGHTED", "1") == "1"
# ε-graph (RELATIVE margin): keep a non-nearest kNN edge only if its similarity is within GNN_KNN_EPS
# of that leaf's OWN nearest similarity (the single nearest is always kept → min-degree 1; KNN_K caps
# degree).  Degree ADAPTS — loners keep ~1 edge, tight clades up to k — fixing the fixed-k over-balance
# (worst at small S) without per-S tuning (relative, so size/diversity-scale-free).  0 = off (top-k).
# Starting sweep ~0.05–0.15 (similarity units); smaller = stricter (fewer edges).
KNN_EPS     = float(os.environ.get("GNN_KNN_EPS", "0.0"))
# Gradient-checkpoint the per-timestep MPNN encode to fit large configs (high KNN_K / SEQ_LEN /
# size) without cutting batch — recomputes encode activations in backward (~25-40% slower, EXACT
# same gradients/accuracy).  GNN_GRAD_CKPT=1 to enable.
GRAD_CKPT   = os.environ.get("GNN_GRAD_CKPT", "0") == "1"
TREE_EDGE_VAL = 1.0      # legacy edge_attr value (all-ones) → tree/scaffold edges
KNN_EDGE_VAL  = -1.0     # distinct value so the MPNN can tell a kNN shortcut from a tree edge
# haplotype window namespaces the checkpoint (changes input dim); appended only for k>0
# so the legacy single-SNP name stays gnn_v7_varsize_best.pt.
_HAP_TAG    = f"_w{HAP_K}" if HAP_K > 0 else ""
_RK_TAG     = (f"_rk{RANK_WEIGHT}" + (f"L{RANK_LEVELS}" if RANK_LEVELS > 1 else "")) if RANK_WEIGHT > 0 else ""
_JK_TAG     = f"_jk{JK_MODE}" if JK_MODE else ""
_KNN_TAG    = (f"_knn{KNN_K}" + ("w" if KNN_WEIGHTED else "") + (f"eps{KNN_EPS}" if KNN_EPS > 0 else "")) if KNN_K > 0 else ""
_CAS_TAG    = (f"_cas-{STAGE1_TAG}" + (f"_anc{ANCHOR_WEIGHT}" if ANCHOR_WEIGHT > 0 else "")) if VS_INPUT == "cascade" else ""
# GNN_TAG: free-form USER label to distinguish checkpoints with identical params but a difference the
# user has in mind (e.g. a data version).  Blank by default → legacy name; else appended as ..._best_<TAG>.pt.
_USER_TAG   = os.environ.get("GNN_TAG", "").strip()
_USER_TAG   = f"_{_USER_TAG}" if _USER_TAG else ""
SAVE_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"gnn_v7_varsize{_HAP_TAG}{_RK_TAG}{_JK_TAG}{_KNN_TAG}{_CAS_TAG}_best{_USER_TAG}.pt")
# Final per-size eval uses SLIDING-WINDOW (full-context) inference by default: each tree is
# predicted from the window where it sits in the central core [margin, SEQ_LEN-margin), which
# removes the bidirectional-GRU edge/boundary artefact (~half the trees at long SEQ_LEN sit in
# the truncated-context edge region).  GNN_SLIDE_MARGIN=0 → legacy non-overlapping eval.
SLIDE_MARGIN = int(os.environ.get("GNN_SLIDE_MARGIN", str(SEQ_LEN // 4)))
EVAL_STRIDE  = int(os.environ.get("GNN_EVAL_STRIDE", "4"))


# ── variable-size input pipeline ────────────────────────────────────────────────
# The size-parametric topology helpers (tskit_tree_to_pyg, tree_cophenetic,
# createSequenceFromTrees, local_upgma_edges, cophenetic_from_parent) are imported
# from gnn_topology; what remains here is the trainer-specific input construction
# (genotype-kNN shortcut edges, cascade scaffolds, sliding-window eval).
def _knn_leaf_edges(geno, S, k, eps=0.0):
    """Undirected leaf↔leaf edges with per-edge genotype SIMILARITY (1 − hamming ∈ [0,1]) as weight.
    geno: [S, T] (±1) focal SNPs over the window (same metric as the UPGMA input).  Returns
    (edge_index [2,2E], sim [2E,1]); symmetric + de-duplicated, both directions carry the same sim.

    eps=0 → plain top-k (each leaf to its k nearest).  eps>0 → ε-GRAPH with a RELATIVE margin: keep the
    single nearest (min-degree 1) plus any neighbour whose similarity is within `eps` of that leaf's OWN
    nearest similarity (sim ≥ best_i − eps).  Degree then ADAPTS — a "loner" (decreasing sims) keeps ~1
    edge, a tight clade (near-tied sims) keeps up to k — removing the fixed-k forced over-clustering that
    over-balances the tree (worst at small S).  RELATIVE not absolute because median kNN similarity rises
    with S (~0.84→0.99 from S20→S100), so an absolute cutoff is badly size-scaled; a per-leaf margin is
    scale-free.  k stays the degree CAP."""
    k = min(k, S - 1)
    if k <= 0:
        return torch.empty(2, 0, dtype=torch.long), torch.empty(0, 1)
    D = squareform(pdist(geno.astype(float), metric="hamming"))     # [S, S] hamming ∈ [0,1]
    np.fill_diagonal(D, np.inf)
    nbr = np.argsort(D, axis=1)[:, :k]                              # [S, k] nearest leaves, rank-ordered
    pairs = {}                                                     # (a,b) → similarity, deduped
    for i in range(S):
        best = 1.0 - float(D[i, nbr[i][0]])                        # leaf i's nearest similarity
        for rank, j in enumerate(nbr[i]):
            j = int(j)
            if i == j:
                continue
            sim = 1.0 - float(D[i, j])
            if eps > 0.0 and rank > 0 and sim < best - eps:         # ε-graph: keep only within margin of best
                continue                                            # (rank 0 = nearest, always kept)
            a, b = (i, j) if i < j else (j, i)
            pairs[(a, b)] = sim
    if not pairs:
        return torch.empty(2, 0, dtype=torch.long), torch.empty(0, 1)
    ab = list(pairs.keys()); sims = [pairs[p] for p in ab]
    a = [p[0] for p in ab]; b = [p[1] for p in ab]
    ei = torch.tensor([a + b, b + a], dtype=torch.long)            # undirected
    w = torch.tensor(sims + sims, dtype=torch.float).unsqueeze(1)  # [2E,1], same both directions
    return ei, w


def local_input_graph(geno, S):
    """Build one window's input graph.  Returns (tree_ei, model_ei, model_ea):
      - tree_ei  : pure local-UPGMA tree edges (used for the input-baseline RF, which needs a tree)
      - model_ei : the graph the MODEL sees = tree edges (+ optional genotype-kNN shortcut edges)
      - model_ea : matching 1-dim edge feature — tree = +TREE_EDGE_VAL; kNN = −similarity
                   (weighted, default) or KNN_EDGE_VAL (flat).  Sign encodes type, magnitude strength.
    KNN_K=0 → model_ei == tree_ei and model_ea is all-ones (byte-identical to the legacy path)."""
    tree_ei = local_upgma_edges(geno, S).long()
    if KNN_K <= 0:
        return tree_ei, tree_ei, torch.full((tree_ei.shape[1], 1), TREE_EDGE_VAL)
    knn_ei, knn_sim = _knn_leaf_edges(geno, S, KNN_K, eps=KNN_EPS)
    model_ei = torch.cat([tree_ei, knn_ei], dim=1)
    knn_ea = -knn_sim if KNN_WEIGHTED else torch.full((knn_ei.shape[1], 1), KNN_EDGE_VAL)
    model_ea = torch.cat([torch.full((tree_ei.shape[1], 1), TREE_EDGE_VAL), knn_ea], dim=0)
    return tree_ei, model_ei, model_ea


def build_dataset(seqs, S):
    dataset = []
    for st in seqs:
        # focal SNP = window centre (col HAP_K); shape [S, T] over the window's positions
        geno = np.stack([x[:S, HAP_K].numpy() for _, _, x, _ in st], axis=1)
        _, ei, ea = local_input_graph(geno, S)
        graphs = [Data(x=x.float(), edge_index=ei, edge_attr=ea) for _, _, x, _ in st]
        true_M = torch.stack([m for *_, m in st], dim=0)                    # [T, S, S] normalised
        dataset.append((graphs, {"true_M": true_M}))
    return dataset


def collate(samples):
    steps = [Batch.from_data_list([s[0][t] for s in samples]) for t in range(SEQ_LEN)]
    true_M = torch.stack([s[1]["true_M"] for s in samples])
    base = (torch.stack([s[1]["baseline_M"] for s in samples])
            if "baseline_M" in samples[0][1] else None)      # [B, T, S, S] residual baseline
    return steps, true_M, base


def load_size_files(scenario, size, file_indices):
    seqs = []
    for i in file_indices:
        p = os.path.join(DATA_DIR, f"{scenario}_S{size}_{i}.trees")
        if os.path.exists(p):
            seqs += createSequenceFromTrees(tskit.load(p), size)
    return seqs


def _parent_to_edge_index(parent, N):
    """parent_index [N] (root=-1) → undirected edge_index [2,2E] (src=parent,dst=child)+reverse."""
    src, dst = [], []
    for c in range(N):
        pa = int(parent[c])
        if pa >= 0:
            src.append(pa); dst.append(c)
    return torch.tensor([src + dst, dst + src], dtype=torch.long)


def _cache_path(scenario, S, i):
    return os.path.join(DATA_DIR, f"{scenario}_S{S}_{i}_stage1_{STAGE1_TAG}.npz")


def _scaffold_eis(scenario, S, i, n_expected):
    """Per-position scaffold edge_index list for file i (cascade), from the cached stage-1
    parents, aligned to _flat_stream order.  None if the cache is absent OR has FEWER positions
    than the file (a STALE cache — e.g. the data was regenerated after the cache was built)."""
    if not os.path.exists(_cache_path(scenario, S, i)):
        return None
    parents = np.load(_cache_path(scenario, S, i))["parents"]          # [n_pos, N]
    if len(parents) < n_expected:                                      # stale / incomplete cache
        return None
    N = 2 * S - 1
    return [_parent_to_edge_index(parents[k], N) for k in range(n_expected)]


def load_size_cascade(scenario, size, file_indices):
    """Stage-2 training dataset with stage-1 scaffolds as input topology.  Window w position t
    uses the cached parent at global index w*SEQ_LEN+t (createSequenceFromTrees and the cache
    share iteration order, so they align by index)."""
    N = 2 * size - 1
    out = []
    for i in file_indices:
        tp = os.path.join(DATA_DIR, f"{scenario}_S{size}_{i}.trees")
        cp = _cache_path(scenario, size, i)
        if not (os.path.exists(tp) and os.path.exists(cp)):
            continue
        seqs = createSequenceFromTrees(tskit.load(tp), size)
        d = np.load(cp)
        parents = d["parents"]
        if len(parents) < len(seqs) * SEQ_LEN:               # STALE cache (data regenerated?) → skip
            print(f"  [skip] {scenario}_S{size}_{i}: stale scaffold cache "
                  f"({len(parents)} positions < {len(seqs) * SEQ_LEN}) — rebuild build_cascade_dataset.py", flush=True)
            continue
        coph = d["coph"] if (ANCHOR_WEIGHT > 0 and "coph" in d) else None   # deep-anchor baseline
        for w, st in enumerate(seqs):
            graphs, base, ok = [], [], True
            for t, (_, _, x, _) in enumerate(st):
                gp = w * SEQ_LEN + t
                if gp >= len(parents):
                    ok = False; break
                ei = _parent_to_edge_index(parents[gp], N)
                graphs.append(Data(x=x.float(), edge_index=ei, edge_attr=torch.ones(ei.shape[1], 1)))
                if coph is not None:
                    base.append(torch.from_numpy(coph[gp].astype(np.float32)))
            if ok:
                tgt = {"true_M": torch.stack([m for *_, m in st], dim=0)}
                if coph is not None:
                    tgt["baseline_M"] = torch.stack(base, dim=0)     # [T, S, S]
                out.append((graphs, tgt))
    return out


def decode_model(M, S, alleles=None):
    """Decode a predicted matrix to a parent-index tree.  DECODER: "nj" | "avg" | "avg_constr" | "nj_constr".
    "avg_constr"/"nj_constr" force the focal SNP's bipartition to be an edge (need `alleles` = focal genotype
    per leaf); they share the allele gate and differ only in balance (UPGMA vs NJ).  Fall back to plain
    "avg" if alleles is None (so callers that don't thread it still work)."""
    N = 2 * S - 1
    if DECODER == "nj":
        return decode_tree_nj(M, num_samples=S, num_nodes=N)
    if DECODER == "nj_constr" and alleles is not None:
        return decode_tree_nj_constrained(M, alleles, num_samples=S, num_nodes=N)
    if DECODER == "avg_constr" and alleles is not None:
        return decode_tree_constrained(M, alleles, num_samples=S, num_nodes=N)
    return decode_tree_from_mrca(M, num_samples=S, num_nodes=N)


def tree_to_parent_clade(tree, S):
    """Topology of a marginal tree (e.g. Relate) → parent_index in our clade-labeled
    convention (leaves 0..S-1; internals ordered by (clade size, sorted leaves))."""
    N = 2 * S - 1
    node_map = {s: s for s in range(S)}
    internals = [n for n in tree.nodes() if not tree.is_leaf(n)]
    def key(n):
        leaves = tuple(sorted(tree.leaves(n)))
        return (len(leaves), leaves)
    for idx, n in enumerate(sorted(internals, key=key)):
        node_map[n] = S + idx
    parent = np.full(N, -1, dtype=np.int64)
    for child in tree.nodes():
        p = tree.parent(child)
        if p != -1 and child in node_map and p in node_map:
            parent[node_map[child]] = node_map[p]
    return torch.from_numpy(parent)


def tskit_marginal_newick(tree, S):
    """Polytomy-safe newick of a marginal tree (e.g. tsinfer), leaves labeled 0..S-1 to match
    parent_index_to_newick's taxa, for RF vs the true tree.  Built straight from the tskit tree, so
    it handles NON-binary trees — unlike tree_to_parent_clade, which assumes the binary 2S-1 layout
    and silently mangles polytomous trees (gaps → spurious roots).  Returns None if not single-rooted."""
    if tree.num_roots != 1:
        return None
    return tree.as_newick(node_labels={u: str(u) for u in tree.samples()})


# ── Triplet topology distance (graded, non-saturating alternative to RF) ─────────
TRIPLET_SAMP = int(os.environ.get("GNN_TRIPLET_SAMP", "3000"))   # sampled triples per size when C(S,3) is large
_TRIP_RNG    = np.random.default_rng(12345)


def _triple_indices(S, max_enum=25000):
    """(I,J,K) index arrays over leaf triples i<j<k — exact enumeration when C(S,3) is small,
    else a fixed Monte-Carlo sample (seeded, computed ONCE per size).  None if S<3."""
    if S < 3:
        return None
    from math import comb
    from itertools import combinations
    if comb(S, 3) <= max_enum:
        tr = np.fromiter((x for t in combinations(range(S), 3) for x in t), dtype=np.int64)
        tr = tr.reshape(-1, 3)
    else:
        cand = np.sort(_TRIP_RNG.integers(0, S, size=(TRIPLET_SAMP * 3, 3)), axis=1)
        cand = cand[(cand[:, 0] < cand[:, 1]) & (cand[:, 1] < cand[:, 2])]
        tr = cand[np.sort(np.unique(cand, axis=0, return_index=True)[1])][:TRIPLET_SAMP]   # first draws, not smallest
    return tr[:, 0], tr[:, 1], tr[:, 2]


def _triplet_labels(M, tI, tJ, tK):
    """Rooted-triplet topology per triple from a cophenetic matrix M (smaller value = more recent
    MRCA = the cherry): 0=cherry(i,j), 1=cherry(i,k), 2=cherry(j,k), 3=unresolved (tie/polytomy)."""
    M = M.detach().cpu().numpy() if torch.is_tensor(M) else np.asarray(M)
    v = np.stack([M[tI, tJ], M[tI, tK], M[tJ, tK]], axis=-1)
    mn = v.min(axis=-1, keepdims=True)
    lab = v.argmin(axis=-1)
    return np.where((v == mn).sum(axis=-1) == 1, lab, 3)


def wrong_triplet_fraction(true_labels, M, tI, tJ, tK):
    """Proportion of sampled triples whose induced topology differs from the true tree's."""
    return float((true_labels != _triplet_labels(M, tI, tJ, tK)).mean())


def build_position_windows(ts):
    """Per-window [SEQ_LEN] genomic positions, chunked identically to createSequenceFromTrees."""
    positions = [v.site.position for v in ts.variants()]
    cur, out = [], []
    for tree in ts.trees():
        l = bisect.bisect_left(positions, tree.interval.left)
        r = bisect.bisect_left(positions, tree.interval.right)
        if l == r:
            continue
        for k in range(l, r):
            cur.append(positions[k])
            if len(cur) == SEQ_LEN:
                out.append(cur); cur = []
    return out


# ── sliding-window (full-context) inference for eval ─────────────────────────────
def _flat_stream(ts, S):
    """Ordered per-position items across the WHOLE file (not chunked):
    (x [N,W] node features, M [S,S] true cophenetic, genomic position)."""
    variants  = list(ts.variants())
    positions = [v.site.position for v in variants]
    G = encode_genotype_matrix(variants)
    items = []
    for tree in ts.trees():
        l = bisect.bisect_left(positions, tree.interval.left)
        r = bisect.bisect_left(positions, tree.interval.right)
        if l == r:
            continue
        M = tree_cophenetic(tree, S)
        for vi in range(l, r):
            _, _, x = tskit_tree_to_pyg(tree, hap_window(G, vi), S)
            items.append((x, M, positions[vi]))
    return items


def _run_window(model, xs, S, device, eis=None, embed_space=None):
    """Run the model on one window (list of node-feature tensors).  Input topology is the
    local-UPGMA tree from the window's focal genotypes (eis=None), OR a per-position scaffold
    edge_index list `eis` (cascade).  Returns (pred [len,S,S], per-position edge_index list).
    embed_space in {None,'L','P','both'}: also return per-position leaf embeddings as a 3rd element,
    a dict {'L': [len,S,H] contextual, 'P': [len,S,D] metric space (distances=tree)} (requested keys)."""
    if eis is None:                                          # local-greedy: one UPGMA per window
        geno = np.stack([x[:S, HAP_K].numpy() for x in xs], axis=1)
        tree_ei, model_ei, model_ea = local_input_graph(geno, S)
        eis = [tree_ei] * len(xs)                            # pure tree → baseline RF unaffected
        gea = [(model_ei, model_ea)] * len(xs)               # what the model sees (maybe kNN-augmented)
    else:                                                    # cascade scaffold: tree edges, no kNN
        gea = [(e, torch.full((e.shape[1], 1), TREE_EDGE_VAL)) for e in eis]
    batched = [Batch.from_data_list(
                   [Data(x=x, edge_index=gea[j][0], edge_attr=gea[j][1])]).to(device)
               for j, x in enumerate(xs)]
    out = model(batched)
    pred = out["mrca"][0].cpu()
    if embed_space:
        emb = {}
        if embed_space in ("L", "both"):
            if "leaf_embed" not in out:
                raise RuntimeError("checkpoint does not expose 'leaf_embed' (needs a pairwise_mrca model)")
            emb["L"] = out["leaf_embed"][0].cpu()           # [T,S,H]
        if embed_space in ("P", "both"):
            if "leaf_embed_metric" not in out:
                raise RuntimeError("checkpoint does not expose 'leaf_embed_metric' (update model / re-export)")
            emb["P"] = out["leaf_embed_metric"][0].cpu()    # [T,S,D]
        return pred, eis, emb
    return pred, eis


def _sliding_predict(model, items, S, device, window=None, margin=None, scaffold_eis=None, embed_space=None):
    """Best-context prediction for every position: slide a window (stride=core=window-2*margin)
    and keep each position's prediction ONLY from the window where it lands in the central core
    [margin, window-margin).  Every interior tree is then evaluated with full bidirectional
    context.  `scaffold_eis` (cascade): per-position input edge_index aligned to `items`; None →
    local-UPGMA per window.  Returns (preds[gp], input_ei[gp]).  margin=0 → non-overlapping.
    embed_space in {None,'L','P','both'} → also returns (embeds, win_ids): embeds is a dict
    {space: [n] list of per-position [S,·] tensors}, win_ids[gp] = the forward-window each pos's
    embedding came from.  Kept from the SAME best-context window as the prediction."""
    if window is None:
        window = SEQ_LEN
    if margin is None:
        margin = SLIDE_MARGIN
    n = len(items)
    preds, in_eis = [None] * n, [None] * n
    spaces = (["L", "P"] if embed_space == "both" else [embed_space]) if embed_space else []
    embeds = {sp: [None] * n for sp in spaces}
    win_ids = [-1] * n if embed_space else None              # which forward-window each pos's embed came from
    if n <= window:                                          # file shorter than one window
        eis = scaffold_eis[:n] if scaffold_eis is not None else None
        res = _run_window(model, [it[0] for it in items], S, device, eis, embed_space=embed_space)
        pr, ei_list = res[0], res[1]
        for j in range(n):
            preds[j], in_eis[j] = pr[j], ei_list[j]
            if embed_space:
                win_ids[j] = 0
                for sp in spaces: embeds[sp][j] = res[2][sp][j]
        return (preds, in_eis, embeds, win_ids) if embed_space else (preds, in_eis)
    core = max(1, window - 2 * margin)
    starts = list(range(0, n - window + 1, core))
    if starts[-1] != n - window:
        starts.append(n - window)
    for wid, s0 in enumerate(starts):
        xs = [items[s0 + j][0] for j in range(window)]
        eis = ([scaffold_eis[s0 + j] for j in range(window)]
               if scaffold_eis is not None else None)
        res = _run_window(model, xs, S, device, eis, embed_space=embed_space)
        pr, ei_list = res[0], res[1]
        for j in range(window):
            gp = s0 + j
            in_core = margin <= j < window - margin
            if preds[gp] is None or in_core:                 # core overwrites edge-fill; None fills gaps
                preds[gp], in_eis[gp] = pr[j], ei_list[j]
                if embed_space:
                    win_ids[gp] = wid                        # same forward pass
                    for sp in spaces: embeds[sp][gp] = res[2][sp][j]
    return (preds, in_eis, embeds, win_ids) if embed_space else (preds, in_eis)


# ── eval one size: RF of model / local-UPGMA / Relate vs true (+ corr) ───────────
# Model inference is SLIDING-WINDOW (full-context) by default (GNN_SLIDE_MARGIN; 0=legacy).
# The local-UPGMA "input" baseline is the UPGMA of the same window that produced each kept
# model prediction, so input↔model are a consistent pair.
def eval_size_full(model, scenarios, size, file_indices, device, max_pos=3000):
    """Evaluate one leaf count, pooling over the given `scenarios` (a list of demography names).
    Returns a dict of RAW metric lists so callers can pool across scenarios for free (the
    combined 'ALL' row is just the concatenation of the per-scenario lists)."""
    S, N = size, 2 * size - 1
    acc = {k: [] for k in ("corr_in", "corr_mod", "corr_rel", "corr_tsi", "corr_sgr",
                           "rf_in", "rf_mod", "rf_rel", "rf_tsi", "rf_sgr",
                           "trip_in", "trip_mod", "trip_rel", "trip_tsi", "trip_sgr")}
    trip = _triple_indices(S)                       # (I,J,K) leaf-triple indices, once per size
    n_pos = 0
    with torch.no_grad():
        for scenario in scenarios:
            for i in file_indices:
                tp = os.path.join(DATA_DIR, f"{scenario}_S{size}_{i}.trees")
                if not os.path.exists(tp):
                    continue
                true_ts = tskit.load(tp)
                rp = os.path.join(DATA_DIR, f"{scenario}_S{size}_{i}_relate.trees")
                relate_ts = tskit.load(rp) if os.path.exists(rp) else None
                tp_tsi = os.path.join(DATA_DIR, f"{scenario}_S{size}_{i}_tsinfer.trees")
                tsinfer_ts = tskit.load(tp_tsi) if os.path.exists(tp_tsi) else None
                tp_sgr = os.path.join(DATA_DIR, f"{scenario}_S{size}_{i}_singer.trees")
                singer_ts = tskit.load(tp_sgr) if os.path.exists(tp_sgr) else None
                items = _flat_stream(true_ts, S)
                if not items:
                    continue
                scaf = _scaffold_eis(scenario, S, i, len(items)) if VS_INPUT == "cascade" else None
                if VS_INPUT == "cascade" and scaf is None:
                    print(f"    [skip] {scenario}_S{size}_{i}: missing/STALE scaffold cache "
                          f"(positions != file) — rebuild build_cascade_dataset.py", flush=True)
                    continue
                preds, in_eis = _sliding_predict(model, items, S, device, scaffold_eis=scaf)
                for gp in range(0, len(items), EVAL_STRIDE):
                    xg, tM, pos = items[gp]
                    ntr = parent_index_to_newick(decode_tree_from_mrca(tM, num_samples=S, num_nodes=N))
                    if ntr is None:
                        continue
                    pin = build_parent_index_from_topology(in_eis[gp], num_nodes=N, root_idx=N - 1)
                    nin = parent_index_to_newick(pin)
                    # focal SNP genotype per leaf (centre column of the hap window) for avg_constr decode
                    mod_par = decode_model(preds[gp], S, alleles=xg[:S, HAP_K])
                    nmd = parent_index_to_newick(mod_par)
                    in_coph = cophenetic_from_parent(pin, S)        # reused for corr_in + triplet
                    mod_coph = cophenetic_from_parent(mod_par, S)   # decoded-tree cophenetic (matches RF Mod)
                    if nin:
                        r = rf_distance(nin, ntr);  acc["rf_in"].append(r) if r is not None else None
                    if nmd:
                        r = rf_distance(nmd, ntr);  acc["rf_mod"].append(r) if r is not None else None
                    acc["corr_mod"].append(cophenetic_corr(preds[gp][None, None], tM[None, None]))
                    acc["corr_in"].append(cophenetic_corr(in_coph[None, None], tM[None, None]))
                    if trip is not None:                             # wrong-triplet fraction vs true (0..1)
                        tlab = _triplet_labels(tM, *trip)
                        acc["trip_in"].append(wrong_triplet_fraction(tlab, in_coph, *trip))
                        acc["trip_mod"].append(wrong_triplet_fraction(tlab, mod_coph, *trip))
                    if relate_ts is not None:
                        try:
                            rtree = relate_ts.at(pos)
                            rel_coph = tree_cophenetic(rtree, S)
                            nrl = parent_index_to_newick(tree_to_parent_clade(rtree, S))
                            if nrl:
                                r = rf_distance(nrl, ntr);  acc["rf_rel"].append(r) if r is not None else None
                            acc["corr_rel"].append(cophenetic_corr(rel_coph[None, None], tM[None, None]))
                            if trip is not None:
                                acc["trip_rel"].append(wrong_triplet_fraction(tlab, rel_coph, *trip))
                        except Exception:
                            pass
                    if tsinfer_ts is not None:
                        try:
                            ttree = tsinfer_ts.at(pos)
                            tsi_coph = tree_cophenetic(ttree, S)
                            nts = tskit_marginal_newick(ttree, S)   # polytomy-safe (tsinfer trees are non-binary)
                            if nts:
                                r = rf_distance(nts, ntr);  acc["rf_tsi"].append(r) if r is not None else None
                            acc["corr_tsi"].append(cophenetic_corr(tsi_coph[None, None], tM[None, None]))
                            if trip is not None:
                                acc["trip_tsi"].append(wrong_triplet_fraction(tlab, tsi_coph, *trip))
                        except Exception:
                            pass
                    if singer_ts is not None:
                        try:
                            stree = singer_ts.at(pos)
                            sgr_coph = tree_cophenetic(stree, S)
                            nsg = tskit_marginal_newick(stree, S)   # polytomy-safe
                            if nsg:
                                r = rf_distance(nsg, ntr);  acc["rf_sgr"].append(r) if r is not None else None
                            acc["corr_sgr"].append(cophenetic_corr(sgr_coph[None, None], tM[None, None]))
                            if trip is not None:
                                acc["trip_sgr"].append(wrong_triplet_fraction(tlab, sgr_coph, *trip))
                        except Exception:
                            pass
                    n_pos += 1
                    if n_pos >= max_pos:
                        break
                if n_pos >= max_pos:
                    break
            if n_pos >= max_pos:
                break
    return acc


def _pool(dicts):
    """Concatenate a list of raw-metric dicts (for the combined 'ALL' row)."""
    out = {k: [] for k in ("corr_in", "corr_mod", "corr_rel", "corr_tsi", "corr_sgr",
                           "rf_in", "rf_mod", "rf_rel", "rf_tsi", "rf_sgr",
                           "trip_in", "trip_mod", "trip_rel", "trip_tsi", "trip_sgr")}
    for d in dicts:
        for k in out:
            out[k] += d[k]
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device {device} | train sizes {TRAIN_SIZES} | held-out test sizes {TEST_SIZES} | "
          f"scenarios {SCENARIOS}")
    print(f"hidden={NODE_HIDDEN} gru={GRU_HIDDEN}x{GRU_LAYERS} mp={MP_LAYERS} metric={METRIC_DIM} "
          f"type{TYPE_DIM} decoder={DECODER} gamma={GAMMA} mse_w={MSE_WEIGHT} jk={JK_MODE} "
          f"input={VS_INPUT}{('/'+STAGE1_TAG) if VS_INPUT=='cascade' else ''}{f' +anchor{ANCHOR_WEIGHT}' if ANCHOR_WEIGHT>0 else ''} "
          f"seq_len={SEQ_LEN} hap_window={HAP_W}(k={HAP_K}) rank_w={RANK_WEIGHT}(tau={RANK_TAU},levels={RANK_LEVELS},decay={RANK_DECAY}) "
          f"knn_k={KNN_K}{(' (UPGMA + ' + ('similarity-weighted' if KNN_WEIGHTED else 'flat') + ' kNN' + (f', ε-graph eps={KNN_EPS}' if KNN_EPS > 0 else '') + ' edges)') if KNN_K > 0 else ' (UPGMA edges only)'} "
          f"grad_ckpt={GRAD_CKPT} "
          f"eval_slide_margin={SLIDE_MARGIN}{' (non-overlap)' if SLIDE_MARGIN == 0 else ''}")
    print(f"checkpoint → {os.path.basename(SAVE_PATH)}")

    print("Loading data ...", flush=True)
    load_start = time.time()
    # FILE-level split so held-out eval is leakage-free and we can load paired Relate trees.
    n_test_files = max(1, FILES_PER // 6)
    train_loaders, val_loaders, test_files = {}, {}, {}
    for S in TRAIN_SIZES:
        train_idx = list(range(0, FILES_PER - n_test_files))
        test_files[S] = list(range(FILES_PER - n_test_files, FILES_PER))
        # validation = the held-out files (separate replicates), for per-epoch TE monitoring and the
        # best checkpoint; the final-eval table reads the same files.  cascade keeps the 10% split.
        # Pool ALL scenarios into this size's dataset (homogeneous-size batches, mixed demography).
        if VS_INPUT == "cascade":                            # input topology = stage-1 scaffold
            ds_all = []
            for scen in SCENARIOS:
                ds_all += load_size_cascade(scen, S, train_idx)
            random.shuffle(ds_all)
            ncut = max(1, int(0.1 * len(ds_all)))
            va, tr = ds_all[:ncut], ds_all[ncut:]
        else:                                                # local-greedy (default)
            seqs, va_seqs = [], []
            for scen in SCENARIOS:
                seqs += load_size_files(scen, S, train_idx)
                va_seqs += load_size_files(scen, S, test_files[S])
            random.shuffle(seqs)
            tr_seqs = seqs
            tr, va = build_dataset(tr_seqs, S), build_dataset(va_seqs, S)
        train_loaders[S] = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
        val_loaders[S]   = DataLoader(va, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
        print(f"  S={S}: {len(tr)} train / {len(va)} val seqs (scenarios {SCENARIOS}, "
              f"files {train_idx[0]}..{train_idx[-1]}), val files {test_files[S]}", flush=True)
    for S in TEST_SIZES:
        if S in test_files:
            continue
        test_files[S] = list(range(FILES_PER))
        print(f"  S={S} (HELD OUT): test files 0..{FILES_PER-1}", flush=True)
    print(f"  Loading time: {time.time() - load_start:.1f} s", flush=True)

    model = GraphSequenceCorrector(
        node_feat_dim=NODE_FEAT_DIM, edge_feat_dim=EDGE_FEAT_DIM,
        num_nodes=2 * max(TRAIN_SIZES + TEST_SIZES) - 1, num_samples=max(TRAIN_SIZES + TEST_SIZES),
        node_hidden=NODE_HIDDEN, num_mp_layers=MP_LAYERS, gru_hidden=GRU_HIDDEN,
        num_gru_layers=GRU_LAYERS, parent_hidden=32, node_id_dim=0, type_dim=TYPE_DIM,
        topology_only=True, pairwise_mrca=True, metric_dim=METRIC_DIM, jk_mode=JK_MODE,
        grad_checkpoint=GRAD_CKPT,
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    def add_parts(by_size, S, loss, parts):
        """Add one batch's loss and its unweighted parts (1 - Pearson, MSE, rank) to the sums of size S."""
        acc = by_size.setdefault(S, [0.0, 0.0, 0.0, 0.0, 0])
        for k, v in enumerate([loss] + parts):
            acc[k] += v.item() if torch.is_tensor(v) else float(v)
        acc[4] += 1

    def parts_text(acc):
        """Mean loss and mean parts of one size, from the sums of add_parts."""
        return (f"{acc[0] / acc[4]:.4f} (1-pearson {acc[1] / acc[4]:.4f}, mse {acc[2] / acc[4]:.4f}, "
                f"rank {acc[3] / acc[4]:.4f})")

    def run_val():
        model.eval()
        tot, ncorr, n = 0.0, 0.0, 0
        by_size = {}
        with torch.no_grad():
            for S, ld in val_loaders.items():
                for steps, true_M, base in ld:
                    steps = [b.to(device) for b in steps]; true_M = true_M.to(device)
                    base = base.to(device) if base is not None else None
                    pred = model(steps)["mrca"]
                    parts = [cophenetic_corr_loss(pred, true_M, gamma=GAMMA), cophenetic_mse(pred, true_M, gamma=GAMMA), 0.0]
                    loss = parts[0] + MSE_WEIGHT * parts[1]
                    if RANK_WEIGHT > 0:
                        parts[2] = cophenetic_rank_loss(
                            pred, true_M, tau=RANK_TAU, levels=RANK_LEVELS, level_decay=RANK_DECAY)
                        loss = loss + RANK_WEIGHT * parts[2]
                    if ANCHOR_WEIGHT > 0 and base is not None:
                        loss = loss + ANCHOR_WEIGHT * cophenetic_anchor_loss(pred, base)
                    tot += loss.item(); ncorr += cophenetic_corr(pred, true_M); n += 1
                    add_parts(by_size, S, loss, parts)
        return ((tot / n, ncorr / n) if n else (float("nan"), float("nan"))) + (by_size,)

    print(f"\n{'Ep':>3} | {'TR loss':>8} {'TR corr':>7} | {'TE loss':>8} {'TE corr':>7}")
    print("─" * 46)
    print(f"      per size: loss (1-pearson, mse, rank), loss = 1-pearson + {MSE_WEIGHT:g} * mse + "
          f"{RANK_WEIGHT:g} * rank", flush=True)
    best_te = float("inf")
    train_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        # Interleave the per-size loaders in random order WITHOUT materialising every collated batch
        # at once — that held ≈a second copy of the whole dataset in host RAM (CPU OOM at large
        # SEQ_LEN, independent of GPU/grad-ckpt).  Shuffle only the ORDER of draws; batches are pulled
        # lazily, one at a time, so peak host RAM is (dataset + a single batch).
        iters = {S: iter(ld) for S, ld in train_loaders.items()}
        schedule = [S for S, ld in train_loaders.items() for _ in range(len(ld))]
        random.shuffle(schedule)
        tot, ncorr, n = 0.0, 0.0, 0
        by_size, batch_time, lr, epoch_start = {}, 0.0, opt.param_groups[0]["lr"], time.time()
        for S in schedule:
            batch_start = time.time()
            steps, true_M, base = next(iters[S])
            batch_time += time.time() - batch_start
            steps = [b.to(device) for b in steps]; true_M = true_M.to(device)
            base = base.to(device) if base is not None else None
            pred = model(steps)["mrca"]
            parts = [cophenetic_corr_loss(pred, true_M, gamma=GAMMA), cophenetic_mse(pred, true_M, gamma=GAMMA), 0.0]
            loss = parts[0] + MSE_WEIGHT * parts[1]
            if RANK_WEIGHT > 0:
                parts[2] = cophenetic_rank_loss(
                    pred, true_M, tau=RANK_TAU, levels=RANK_LEVELS, level_decay=RANK_DECAY)
                loss = loss + RANK_WEIGHT * parts[2]
            if ANCHOR_WEIGHT > 0 and base is not None:
                loss = loss + ANCHOR_WEIGHT * cophenetic_anchor_loss(pred, base)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
            tot += loss.item(); ncorr += cophenetic_corr(pred.detach(), true_M); n += 1
            add_parts(by_size, S, loss, parts)
        train_time = time.time() - epoch_start
        sched.step()
        te_loss, te_corr, te_by_size = run_val()
        flag = ""
        if te_loss < best_te:
            best_te = te_loss
            save_flushed(model, SAVE_PATH, verbose=False)   # subnormal-free artifact; live model untouched
            flag = " ✓"
        print(f"{epoch:>3} | {tot/n:>8.4f} {ncorr/n:>7.4f} | {te_loss:>8.4f} {te_corr:>7.4f}{flag}", flush=True)
        for S in sorted(by_size):
            va = parts_text(te_by_size[S]) if S in te_by_size else "n/a"
            print(f"      S={S:<4} TR {parts_text(by_size[S])} | VA {va}", flush=True)
        epoch_time = time.time() - epoch_start
        print(f"      time {epoch_time:.1f} s: training {train_time:.1f} s (building batches {batch_time:.1f} s), "
              f"validation and saving {epoch_time - train_time:.1f} s | lr {lr:.4e}", flush=True)

    total_train_time = time.time() - train_start
    print(f"\nTotal training time: {total_train_time:.1f} s "
          f"({total_train_time / 60:.1f} min) over {EPOCHS} epochs", flush=True)

    table_start = time.time()
    print(f"\n── Per-scenario / per-size evaluation — SLIDING-WINDOW full-context inference "
          f"(margin={SLIDE_MARGIN}; corr higher=better, RF lower=better, scales with S) ──")
    # RF improvement of the model vs a reference (positive = model has lower RF = better).
    def pct(mod, ref):
        ok = (mod == mod) and (ref == ref) and ref > 0   # both non-NaN, ref>0
        return f"{(1 - mod / ref) * 100:+.1f}%" if ok else "n/a"
    def mean(x):
        return float(np.mean(x)) if x else float("nan")
    hdr = (f"  {'Size':>5}  {'set':>8} | {'Corr In':>7} {'Corr Mod':>8} {'Corr Rel':>8} {'Corr Tsi':>8} {'Corr Sgr':>8}"
           f" | {'RF In':>7} {'RF Mod':>7} {'RF Rel':>7} {'RF Tsi':>7} {'RF Sgr':>7}"
           f" | {'Mod vs In':>9} {'Mod vs Rel':>10} {'Mod vs Tsi':>10} {'Mod vs Sgr':>10}"
           f" | {'WrTrip In':>9} {'WrTrip Mod':>10} {'WrTrip Rel':>10} {'WrTrip Tsi':>10} {'WrTrip Sgr':>10}")
    def print_row(S, tag, d):
        ci, cm, cr, ct, cs = (mean(d["corr_in"]), mean(d["corr_mod"]), mean(d["corr_rel"]),
                              mean(d["corr_tsi"]), mean(d["corr_sgr"]))
        ri, rm, rr, rt, rs = (mean(d["rf_in"]), mean(d["rf_mod"]), mean(d["rf_rel"]),
                              mean(d["rf_tsi"]), mean(d["rf_sgr"]))
        ti, tm, tr, tt, ts = (mean(d["trip_in"]), mean(d["trip_mod"]), mean(d["trip_rel"]),
                              mean(d["trip_tsi"]), mean(d["trip_sgr"]))
        print(f"  {S:>5}  {tag:>8} | {ci:>7.4f} {cm:>8.4f} {cr:>8.4f} {ct:>8.4f} {cs:>8.4f}"
              f" | {ri:>7.2f} {rm:>7.2f} {rr:>7.2f} {rt:>7.2f} {rs:>7.2f}"
              f" | {pct(rm, ri):>9} {pct(rm, rr):>10} {pct(rm, rt):>10} {pct(rm, rs):>10}"
              f" | {ti:>9.4f} {tm:>10.4f} {tr:>10.4f} {tt:>10.4f} {ts:>10.4f}", flush=True)

    if os.path.exists(SAVE_PATH):                       # evaluate the BEST (lowest TE loss) model
        model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    model.eval()
    # raw[scenario][S] = metric-list dict, so we can pool for the combined 'ALL' block for free.
    raw = {scen: {} for scen in SCENARIOS}
    for scen in SCENARIOS:
        print(f"\n=== scenario: {scen} ===")
        print(hdr)
        any_data = False
        for S in sorted(test_files):
            tag = "held-out" if S in TEST_SIZES and S not in TRAIN_SIZES else "in-dist"
            d = eval_size_full(model, [scen], S, test_files[S], device)
            raw[scen][S] = d
            if d["corr_mod"]:
                print_row(S, tag, d); any_data = True
        if not any_data:
            print("  (no files for this scenario)")
    if len(SCENARIOS) > 1:
        print(f"\n=== ALL scenarios pooled ===")
        print(hdr)
        for S in sorted(test_files):
            tag = "held-out" if S in TEST_SIZES and S not in TRAIN_SIZES else "in-dist"
            d = _pool([raw[scen][S] for scen in SCENARIOS])
            if d["corr_mod"]:
                print_row(S, tag, d)
    print(f"\nEnd table time: {time.time() - table_start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
