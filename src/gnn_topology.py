"""
gnn_topology.py — shared library for the ARG topology corrector
================================================================
The size-agnostic core shared by the trainer (train.py) and every diagnostic
script.  It holds three things that were previously tangled into the trainer:

  1. Config / hyperparameters — model architecture + loss knobs, read from the
     environment once at import (SEQ_LEN, HAP_K, hidden sizes, MP depth, ranking
     loss weights, …).  The trainer imports these; consumers read them via the
     trainer's namespace.
  2. Losses & metrics — the cophenetic (leaf-pair MRCA-clade-size) regression
     objective and its correlation/rank/anchor variants.  All permutation- and
     size-invariant, so they live here rather than in a size-specific file.
  3. Topology helpers & decoders — SIZE-PARAMETRIC (every function takes the leaf
     count S explicitly): tskit tree → PyG graph, tree → cophenetic matrix,
     UPGMA/NJ decode of a predicted matrix, newick + Robinson–Foulds.

Topology represented purely by the cophenetic matrix
  M[i, j] = size of the smallest clade containing leaves i and j  (∈ 2..S),
NORMALISED by S so it is comparable across leaf counts.  This is ultrametric,
uniquely determines the topology, and is stable over internal-node relabeling
and along the sequence — i.e. a well-posed regression target (unlike per-node
parent slots, which have no stable identity; see the project journal).
"""

import bisect
import os

import dendropy
from dendropy.calculate import treecompare
import numpy as np
import torch
import tskit
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

# ── Config / hyperparameters (read once at import) ──────────────────────────────
# Sequence chunking: trees are streamed into fixed-length windows of SEQ_LEN
# consecutive variant positions (the GRU's temporal context).
SEQ_LEN        = int(os.environ.get("GNN_SEQ_LEN", "64"))
# Per-leaf haplotype window: each leaf's alleles at the focal SNP + HAP_K flanking
# SNPs each side (W = 2*HAP_K+1), sliced from the whole-file genotype matrix so it
# reads across SEQ_LEN chunk boundaries and zero-pads only at file ends.  HAP_K=0 →
# single focal SNP (legacy).  Injects the cherry/sibling-sharing signal a lone SNP lacks.
HAP_K          = int(os.environ.get("GNN_HAP_WINDOW", "0"))
HAP_W          = 2 * HAP_K + 1
NODE_FEAT_DIM  = HAP_W
# Node feature identity: leaf/internal TYPE embedding (permutation-equivariant,
# size-agnostic) is the default; GNN_TYPE_DIM=0 falls back to a per-node IDENTITY
# embedding (fixed-size only).
TYPE_DIM       = int(os.environ.get("GNN_TYPE_DIM", "8"))
NODE_ID_DIM    = 0 if TYPE_DIM > 0 else 32
EDGE_FEAT_DIM  = 1
# Model capacity (defaults reproduce the working v5 GRU; raise for scale-up tests).
NODE_HIDDEN    = int(os.environ.get("GNN_HIDDEN", "64"))
GRU_HIDDEN     = int(os.environ.get("GNN_GRU_HIDDEN", "64"))
GRU_LAYERS     = int(os.environ.get("GNN_GRU_LAYERS", "1"))
MP_LAYERS      = int(os.environ.get("GNN_MP_LAYERS", "3"))    # MPNN depth (cross-node hops)
METRIC_DIM     = int(os.environ.get("GNN_METRIC_DIM", "32"))
# Cherry-weighting: weight each leaf pair by (true clade size)^(-GAMMA) in the loss.
# GAMMA=0 → uniform; larger → more weight on recent coalescences (cherries), which
# RF cares about and the correlation loss ignores.
GAMMA          = float(os.environ.get("GNN_GAMMA", "0.0"))
# Ranking (nearest-coalescence) loss — pushes each leaf's predicted-distance minimum
# onto its TRUE nearest neighbour(s), optimising the ORDER the decoder reads (which
# pair joins first = cherries), which corr/MSE are blind to.  NORMALISED (KL, ÷log S)
# to sit in ~[0,1] like (1-corr): RANK_WEIGHT≈1 is "on par with corr", 0 → off.
RANK_WEIGHT    = float(os.environ.get("GNN_RANK_WEIGHT", "0.0"))
RANK_TAU       = float(os.environ.get("GNN_RANK_TAU", "1.0"))
# Generalised ranking: enforce the coalescence ORDER over the first GNN_RANK_LEVELS
# shallow levels (peeled Plackett–Luce), not just the nearest (=1).  Levels weighted
# GNN_RANK_DECAY**(m-1) (shallow emphasis).
RANK_LEVELS    = int(os.environ.get("GNN_RANK_LEVELS", "1"))
RANK_DECAY     = float(os.environ.get("GNN_RANK_DECAY", "0.5"))
# Jumping-knowledge over MP layers: "" (off, last layer only) | "cat" | "max" | "sum".
# Exposes early-layer (local) features to the head alongside the deep (over-smoothed)
# last layer — counters over-smoothing without dropping MP depth.
JK_MODE        = os.environ.get("GNN_JK", "").lower() or None


# ── Feature helpers ─────────────────────────────────────────────────────────────
def encode_genotype_matrix(variants):
    """Whole-file genotype matrix [S, n_variants], encoded 0(ancestral)→-1, 1(derived)→+1.
    Built once per file so haplotype windows can be sliced across SEQ_LEN chunk boundaries."""
    G = np.asarray([v.genotypes for v in variants], dtype=float).T   # [S, n_var]
    G[G == 0] = -1.0
    return G


def hap_window(G, idx, k=None):
    """[S, 2k+1] haplotype window centred on focal variant `idx`.

    G spans the whole file, so the window reads real flanking alleles across SEQ_LEN
    chunk boundaries (an artificial batching split, not a biological one); it zero-pads
    ONLY at the true file ends.  k=HAP_K by default; k=0 → the single focal SNP (legacy)."""
    if k is None:
        k = HAP_K
    if k == 0:
        return G[:, idx:idx + 1]
    S, n = G.shape
    win = np.zeros((S, 2 * k + 1), dtype=float)
    lo, hi = idx - k, idx + k + 1
    a, b = max(lo, 0), min(hi, n)
    win[:, a - lo:b - lo] = G[:, a:b]
    return win


# ── Size-parametric topology helpers (every function takes the leaf count S) ─────
def tskit_tree_to_pyg(tree, geno_window, S):
    """tskit tree + [S, W] leaf haplotype window (±1, file-end zero-padded) → PyG tensors.
    Internal nodes get a zero row of width W.  Internals are relabeled by clade content
    (size, sorted leaves) so parent index > child index, matching the cophenetic layout."""
    N = 2 * S - 1
    nodes = list(tree.nodes())
    node_map = {s: s for s in range(S)}
    internals = [n for n in nodes if n >= S]
    def key(n):
        leaves = tuple(sorted(tree.leaves(n)))
        return (len(leaves), leaves)
    for idx, n in enumerate(sorted(internals, key=key)):
        node_map[n] = S + idx
    edges = []
    for child in nodes:
        p = tree.parent(child)
        if p != -1:
            edges.append((node_map[p], node_map[child]))
    src, dst = zip(*edges)
    ei = torch.tensor([list(src) + list(dst), list(dst) + list(src)], dtype=torch.long)
    ea = torch.ones(len(src) * 2, 1)
    x = np.concatenate([geno_window, np.zeros((N - S, geno_window.shape[1]))], axis=0)
    x = torch.tensor(x, dtype=torch.float)            # [N, W]
    return ei, ea, x


def tree_cophenetic(tree, S):
    """Symmetric [S, S] matrix M[i,j] = #leaves under MRCA(i,j) (clade size), NORMALISED
    by S → ∈(0,1] so it is comparable across leaf counts.  Single post-order pass: at each
    internal node, every pair of leaves split across two different children has its MRCA there."""
    M = np.zeros((S, S), dtype=np.float32)
    leaves_under = {}
    for n in tree.nodes(order="postorder"):
        kids = tree.children(n)
        if not kids:
            leaves_under[n] = [n] if n < S else []
            continue
        cl = [leaves_under[c] for c in kids]
        tot = [l for c in cl for l in c]; sz = len(tot)
        for a in range(len(cl)):
            for b in range(a + 1, len(cl)):
                La, Lb = cl[a], cl[b]
                if La and Lb:
                    ia, ib = np.array(La), np.array(Lb)
                    M[np.ix_(ia, ib)] = sz; M[np.ix_(ib, ia)] = sz
        leaves_under[n] = tot
    return torch.from_numpy(M / S)            # NORMALISE by S → ∈(0,1], size-comparable


def createSequenceFromTrees(ts, S):
    """Stream a tree sequence into fixed-length windows of SEQ_LEN consecutive variant
    positions.  Each item is (edge_index, edge_attr, node_features, cophenetic_M)."""
    variants = list(ts.variants())
    positions = [v.site.position for v in variants]
    G = encode_genotype_matrix(variants)               # [S, n_var], encoded once per file
    cur, seqs = [], []
    for tree in ts.trees():
        l = bisect.bisect_left(positions, tree.interval.left)
        r = bisect.bisect_left(positions, tree.interval.right)
        if l == r:
            continue
        M = tree_cophenetic(tree, S)
        for vi in range(l, r):                         # global variant index → window
            cur.append((*tskit_tree_to_pyg(tree, hap_window(G, vi), S), M))
            if len(cur) == SEQ_LEN:
                seqs.append(cur); cur = []
    return seqs


def local_upgma_edges(geno, S):
    """Clade-labeled edge_index of the UPGMA tree built from a window's SNPs (hamming
    distance, average linkage).  Relate-free local starting tree; internals labeled by
    (clade size, sorted leaves) so parent index > child index."""
    Z = linkage(pdist(geno.astype(float), metric="hamming"), method="average")
    clade = {i: frozenset([i]) for i in range(S)}
    for k, (a, b, _, _) in enumerate(Z):
        clade[S + k] = clade[int(a)] | clade[int(b)]
    internals = list(range(S, S + len(Z)))
    order = sorted(internals, key=lambda c: (len(clade[c]), tuple(sorted(clade[c]))))
    nodemap = {i: i for i in range(S)}
    for r, c in enumerate(order):
        nodemap[c] = S + r
    edges = []
    for k, (a, b, _, _) in enumerate(Z):
        edges.append((nodemap[S + k], nodemap[int(a)]))
        edges.append((nodemap[S + k], nodemap[int(b)]))
    src, dst = zip(*edges)
    return torch.tensor([list(src) + list(dst), list(dst) + list(src)], dtype=torch.long)


def cophenetic_from_parent(parent, S):
    """Clade-size cophenetic [S,S] from a parent_index (root parent = -1), N=2S-1.
    Values are RAW clade sizes (not normalised)."""
    N = 2 * S - 1
    p = np.asarray(parent.tolist() if torch.is_tensor(parent) else parent).astype(int)
    children = [[] for _ in range(N)]
    root = -1
    for i in range(N):
        if p[i] < 0:
            root = i
        else:
            children[p[i]].append(i)
    size = [0] * N
    def _sz(n):
        if not children[n]:
            size[n] = 1
            return 1
        s = sum(_sz(c) for c in children[n]); size[n] = s
        return s
    _sz(root)
    anc = []
    for leaf in range(S):
        path, x = [], leaf
        while x >= 0:
            path.append(x); x = int(p[x])
        anc.append(path)
    M = np.zeros((S, S), dtype=np.float32)
    for i in range(S):
        ai = set(anc[i])
        for j in range(i + 1, S):
            for x in anc[j]:
                if x in ai:
                    M[i, j] = M[j, i] = size[x]
                    break
    return torch.from_numpy(M)


# ── Loss & metrics (size- and permutation-invariant) ────────────────────────────
def _pair_mask(S, device):
    return torch.triu(torch.ones(S, S, dtype=torch.bool, device=device), diagonal=1)


def _pair_weights(g, gamma):
    """Per-pair weights from TRUE clade sizes g: w = g^(-gamma), normalised to mean 1
    per tree.  gamma=0 → all-ones.  Weights use the target (not the prediction), so they
    are a fixed reweighting the model cannot game."""
    if gamma <= 0:
        return torch.ones_like(g)
    w = g.clamp_min(1.0) ** (-gamma)
    return w / w.mean(dim=-1, keepdim=True).clamp_min(1e-9)


def cophenetic_mse(pred, target, gamma=0.0):
    S = pred.shape[-1]
    m = _pair_mask(S, pred.device)
    p, g = pred[..., m], target[..., m]
    se = (p - g) ** 2
    if gamma <= 0:
        return se.mean()
    return (_pair_weights(g, gamma) * se).mean()


def cophenetic_anchor_loss(pred, baseline, eps=1e-9):
    """Deep-weighted MSE pulling the prediction toward a stage-1 scaffold's cophenetic matrix
    (the cascade "keep the deep structure" term).  Each pair is weighted by the baseline's
    clade size (normalised mean-1 per tree): DEEP pairs anchored strongly, CHERRIES weakly —
    so the rank loss is free to re-resolve them.  Training regulariser only (no baseline at
    inference), which keeps the backbone gradient healthy (unlike a residual head)."""
    S = pred.shape[-1]
    m = _pair_mask(S, pred.device)
    p, b = pred[..., m], baseline[..., m]
    w = b.clamp_min(0.0)
    w = w / w.mean(dim=-1, keepdim=True).clamp_min(eps)        # deep pairs weighted up, mean 1
    return (w * (p - b) ** 2).mean()


def cophenetic_rank_loss(pred, target, tau=1.0, levels=1, level_decay=0.5, eps=1e-9):
    """Normalised coalescence-ORDER ranking loss = mean-over-leaves KL( true level ‖
    softmax(-M_pred/tau) ), divided by log(S).  Optimises the ORDER the decoder reads
    (which pair joins first/next = the topology), which corr/MSE are blind to.

    levels=1 (default): NEAREST-only.  Target per leaf = uniform over its TRUE nearest
    neighbour(s) (its smallest clade; ties shared so a non-cherry leaf defers to its whole
    sibling clade).  Purely ordinal.

    levels>1: a PEELED (Plackett–Luce) generalisation over the first `levels` SHALLOW
    coalescence levels.  At level m the already-assigned closer leaves are masked out and
    the target is uniform over the m-th true level group; extends the ordering pressure from
    cherries down to depth-2,3,… while leaving the already-good deep structure free.  Levels
    weighted level_decay**(m-1) and per-leaf-normalised so RANK_WEIGHT stays interpretable.

    Two normalisations vs a raw softmax-CE: (1) KL = CE − H(target) subtracts the irreducible
    tie-entropy so a perfectly-ordered prediction scores ~0 regardless of ties; (2) ÷log(S)
    is size-fair and ~comparable to (1−corr).  pred, target: [..., S, S].
    """
    S = pred.shape[-1]
    if S < 3:
        return pred.sum() * 0.0                                   # no non-trivial ranking
    eye = torch.eye(S, dtype=torch.bool, device=pred.device)
    neg = torch.finfo(pred.dtype).min
    if levels <= 1:                                               # ── original nearest-only
        logp = torch.log_softmax((-pred / tau).masked_fill(eye, neg), dim=-1)
        t = target.masked_fill(eye, float("inf"))
        rowmin = t.min(dim=-1, keepdim=True).values
        pos = (t <= rowmin + eps) & ~eye
        pcount = pos.sum(dim=-1).clamp_min(1).to(pred.dtype)
        tgt = pos.to(pred.dtype)
        tgt = tgt / tgt.sum(dim=-1, keepdim=True).clamp_min(eps)
        ce = -(tgt * logp).sum(dim=-1)
        kl = (ce - torch.log(pcount)).clamp_min(0.0)
        return kl.mean() / float(np.log(S))
    # ── generalised peeled multi-level ranking ──
    t = target.masked_fill(eye, float("inf"))                    # true dists, self = inf
    peeled = eye.expand(pred.shape).clone()                      # leaves already assigned (start: self)
    total = torch.zeros(pred.shape[:-1], dtype=pred.dtype, device=pred.device)   # [...,S]
    wtot = torch.zeros_like(total)
    for m in range(int(levels)):
        avail = ~peeled                                          # candidates still rankable [...,S,S]
        t_m = t.masked_fill(peeled, float("inf"))
        rowmin = t_m.min(dim=-1, keepdim=True).values
        pos = (t_m <= rowmin + eps) & avail                     # the m-th (next-closest) true level group
        farther = avail & ~pos                                   # leaves that must rank BELOW this group
        valid = farther.any(dim=-1).to(pred.dtype)              # [...,S] only penalise where a real order exists
        cnt = pos.sum(dim=-1).clamp_min(1).to(pred.dtype)
        logp = torch.log_softmax((-pred / tau).masked_fill(peeled, neg), dim=-1)   # softmax over remaining
        tgt = pos.to(pred.dtype)
        tgt = tgt / tgt.sum(dim=-1, keepdim=True).clamp_min(eps)
        ce = -(tgt * logp).sum(dim=-1)
        kl = (ce - torch.log(cnt)).clamp_min(0.0)              # [...,S]
        w = (level_decay ** m) * valid
        total = total + w * kl
        wtot = wtot + w
        peeled = peeled | pos                                    # remove this level before the next
    per_leaf = total / wtot.clamp_min(eps)                       # [...,S] weighted over levels present
    return per_leaf.mean() / float(np.log(S))


def cophenetic_corr(pred, target):
    """Pearson correlation over leaf pairs (higher = better-recovered structure)."""
    S = pred.shape[-1]
    m = _pair_mask(S, pred.device)
    a = pred[..., m].reshape(-1).float()
    b = target[..., m].reshape(-1).float()
    a = a - a.mean(); b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-9)
    return float((a @ b) / denom)


def cophenetic_corr_loss(pred, target, gamma=0.0, eps=1e-6):
    """Per-tree (1 − weighted Pearson) over leaf pairs.

    Scale/offset-invariant, so it rewards matching the *shape* of each tree's cophenetic
    matrix and maximally penalises a constant (position-independent) prediction — directly
    attacking the "collapse to the global mean tree" failure.  gamma>0 weights the Pearson
    toward small clades (cherries); gamma=0 recovers the plain per-tree Pearson."""
    S = pred.shape[-1]
    m = _pair_mask(S, pred.device)
    p = pred[..., m]                                  # [B, T, P]
    g = target[..., m]
    w = _pair_weights(g, gamma)                       # [B, T, P], mean 1 per tree
    wsum = w.sum(-1, keepdim=True).clamp_min(eps)
    pm = (w * p).sum(-1, keepdim=True) / wsum
    gm = (w * g).sum(-1, keepdim=True) / wsum
    pc, gc = p - pm, g - gm
    cov = (w * pc * gc).sum(-1)
    vp  = (w * pc * pc).sum(-1).clamp_min(eps)
    vg  = (w * gc * gc).sum(-1).clamp_min(eps)
    corr = cov / torch.sqrt(vp * vg)                  # [B, T] weighted per-tree Pearson
    return (1.0 - corr).mean()


# ── Topology decode / RF ──────────────────────────────────────────────────────────
def decode_tree_from_mrca(M, num_samples, num_nodes):
    """Average-linkage agglomerative clustering on the cophenetic matrix M → parent index."""
    S = num_samples
    Mnp = M.detach().cpu().numpy().astype(float)
    Mnp = 0.5 * (Mnp + Mnp.T)
    np.fill_diagonal(Mnp, 0.0)
    Z = linkage(squareform(Mnp, checks=False), method="average")
    parent = np.full(num_nodes, -1, dtype=np.int64)
    cl2node = list(range(S))
    for idx, (c1, c2, _, _) in enumerate(Z):
        node = S + idx
        parent[cl2node[int(c1)]] = node
        parent[cl2node[int(c2)]] = node
        cl2node.append(node)
    return torch.from_numpy(parent)


def decode_tree_constrained(M, alleles, num_samples, num_nodes):
    """UPGMA_constrained — average-linkage that GUARANTEES the focal SNP's bipartition is an edge
    of the decoded tree (infinite-sites: derived carriers are monophyletic somewhere in the tree).

    Manual UPGMA (size-weighted average, == decode_tree_from_mrca on a generic matrix), with ONE
    extra rule on which merges are allowed each step:

        A merge whose union spans BOTH focal alleles ("mixed") is permitted only if the union
        already contains ALL of one allele group  (u0 == total0  or  u1 == total1)
        — equivalently, all clusters OUTSIDE the merge are a single allele ("the rest is uniform").

    Why this is correct (and orientation-free): the first allowed mixed merge happens only when one
    allele group is wholly inside it; since every earlier mixed merge was blocked, both merged
    clusters are pure, so that whole group is a pure subtree → a CLADE.  Whichever group finishes
    assembling first becomes the clade; the mutation edge sits at the model-chosen attachment depth
    and the other side stays paraphyletic.  No root-forcing.  alleles: length-S focal genotype per
    leaf (binarised as >0); monomorphic SNP → no constraint → identical to plain UPGMA.  Same
    parent-index output as decode_tree_from_mrca."""
    S, N = num_samples, num_nodes
    M = (M.detach().cpu().numpy().astype(float) if torch.is_tensor(M)
         else np.asarray(M, dtype=float))
    M = 0.5 * (M + M.T)
    grp = (np.asarray(alleles).reshape(-1)[:S] > 0).astype(np.int64)   # 0/1 allele per leaf
    total1 = int(grp.sum()); total0 = S - total1

    D = np.full((N, N), np.inf, dtype=float)
    D[:S, :S] = M
    np.fill_diagonal(D, np.inf)
    size = np.zeros(N, dtype=np.int64)
    cnt0 = np.zeros(N, dtype=np.int64); cnt1 = np.zeros(N, dtype=np.int64)
    size[:S] = 1
    cnt1[:S] = grp; cnt0[:S] = 1 - grp
    parent = np.full(N, -1, dtype=np.int64)
    active = list(range(S))
    nxt = S
    for _ in range(S - 1):
        sub = D[np.ix_(active, active)]          # k x k, symmetric, inf on diagonal
        a0, a1 = cnt0[active], cnt1[active]
        u0 = a0[:, None] + a0[None, :]
        u1 = a1[:, None] + a1[None, :]
        ok = (u0 == 0) | (u1 == 0) | (u0 == total0) | (u1 == total1)
        cand = np.where(ok, sub, np.inf)

        f = cand.argmin()
        if not np.isfinite(cand.flat[f]):        # only pay for the fallback if it fires
            f = sub.argmin()
        i, j = divmod(int(f), len(active))
        if i > j:
            i, j = j, i
        a, b = int(active[i]), int(active[j])

        parent[a] = parent[b] = nxt
        na, nb = size[a], size[b]
        size[nxt] = na + nb
        cnt0[nxt] = cnt0[a] + cnt0[b]
        cnt1[nxt] = cnt1[a] + cnt1[b]

        rest = np.delete(active, (i, j))         # keep `active` as an np.arange(S) array
        newd = (na * D[a, rest] + nb * D[b, rest]) / (na + nb)
        D[nxt, rest] = newd
        D[rest, nxt] = newd
        active = np.append(rest, nxt)
        nxt += 1
    return torch.from_numpy(parent)


def decode_tree_nj(M, num_samples, num_nodes):
    """Neighbour-joining (rooted agglomerative variant) on the predicted matrix.
    Handles non-ultrametric learned distances better than average-linkage."""
    S = num_samples
    D = M.detach().cpu().numpy().astype(float) if torch.is_tensor(M) else np.asarray(M, float)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    dist = {(a, b): D[a, b] for a in range(S) for b in range(a + 1, S)}
    gd = lambda a, b: dist[(a, b)] if a < b else dist[(b, a)]
    parent = np.full(num_nodes, -1, dtype=np.int64)
    active, nxt = list(range(S)), S
    while len(active) > 2:
        m = len(active)
        r = {i: sum(gd(i, k) for k in active if k != i) for i in active}
        best, bq = None, np.inf
        for ai in range(m):
            for bi in range(ai + 1, m):
                i, j = active[ai], active[bi]
                q = (m - 2) * gd(i, j) - r[i] - r[j]
                if q < bq:
                    bq, best = q, (i, j)
        i, j = best
        u, nxt = nxt, nxt + 1
        parent[i] = parent[j] = u
        dij = gd(i, j)
        for k in active:
            if k in (i, j):
                continue
            lo, hi = (u, k) if u < k else (k, u)
            dist[(lo, hi)] = 0.5 * (gd(i, k) + gd(j, k) - dij)
        active.remove(i); active.remove(j); active.append(u)
    i, j = active
    parent[i] = parent[j] = nxt
    return torch.from_numpy(parent)


def decode_tree_nj_constrained(M, alleles, num_samples, num_nodes):
    """Neighbour-joining counterpart of decode_tree_constrained: same focal-SNP allele GATE (so the
    derived carriers stay monophyletic → each site's mutation sits on one branch, homoplasy-free) but the
    pair is chosen by the NJ Q-criterion and distances updated by the NJ reduction instead of UPGMA
    average-linkage.  Only the tree BALANCE differs from avg_constr (NJ under-balances vs UPGMA), so this
    is the A/B knob for testing whether over-balancing drives the deep-node under-aging.  The gate is a
    rule on which UNIONS are legal (set-containment only), independent of the selection criterion, so the
    monophyly guarantee carries over unchanged.  Same parent-index output as decode_tree_constrained."""
    S, N = num_samples, num_nodes
    M = (M.detach().cpu().numpy().astype(float) if torch.is_tensor(M) else np.asarray(M, dtype=float))
    M = 0.5 * (M + M.T)
    grp = (np.asarray(alleles).reshape(-1)[:S] > 0).astype(np.int64)     # 0/1 focal allele per leaf
    total1 = int(grp.sum()); total0 = S - total1

    D = np.zeros((N, N), dtype=float)
    D[:S, :S] = M
    cnt0 = np.zeros(N, dtype=np.int64); cnt1 = np.zeros(N, dtype=np.int64)
    cnt1[:S] = grp; cnt0[:S] = 1 - grp
    parent = np.full(N, -1, dtype=np.int64)
    active, nxt = list(range(S)), S
    for _ in range(S - 1):
        m = len(active)
        r = {i: sum(D[i, k] for k in active if k != i) for i in active}  # NJ row sums over active clusters
        best, bestq = None, np.inf
        fallback, fbq = None, np.inf                                     # closest pair ignoring the gate
        for ai in range(m):
            a = active[ai]
            for bi in range(ai + 1, m):
                b = active[bi]
                q = (m - 2) * D[a, b] - r[a] - r[b]
                if q < fbq:
                    fbq, fallback = q, (a, b)
                u0 = cnt0[a] + cnt0[b]; u1 = cnt1[a] + cnt1[b]
                if u0 > 0 and u1 > 0 and not (u0 == total0 or u1 == total1):
                    continue                                            # forbidden mixed merge (same gate)
                if q < bestq:
                    bestq, best = q, (a, b)
        a, b = best if best is not None else fallback                   # fallback ~never expected to fire
        parent[a] = nxt; parent[b] = nxt
        cnt0[nxt] = cnt0[a] + cnt0[b]; cnt1[nxt] = cnt1[a] + cnt1[b]
        dab = D[a, b]
        for k in active:
            if k == a or k == b:
                continue
            D[nxt, k] = D[k, nxt] = 0.5 * (D[a, k] + D[b, k] - dab)     # NJ distance reduction
        active.remove(a); active.remove(b); active.append(nxt)
        nxt += 1
    return torch.from_numpy(parent)


def parent_index_to_newick(parent_index):
    children = {i: [] for i in range(len(parent_index))}
    root = None
    for i, p in enumerate(parent_index.tolist()):
        if p == -1:
            root = i
        else:
            children[int(p)].append(i)
    if root is None:
        return None

    def _rec(node):
        if not children[node]:
            return str(node)
        return "(" + ",".join(f"{_rec(c)}:1" for c in children[node]) + ")"

    return _rec(root) + ";"


def rf_distance(nwk1, nwk2):
    try:
        tns = dendropy.TaxonNamespace()
        t1 = dendropy.Tree.get(data=nwk1, schema="newick", taxon_namespace=tns,
                               suppress_internal_node_taxa=True)
        t2 = dendropy.Tree.get(data=nwk2, schema="newick", taxon_namespace=tns,
                               suppress_internal_node_taxa=True)
        for t in (t1, t2):
            t.is_rooted = False
            t.collapse_basal_bifurcation()
            t.encode_bipartitions()
        return treecompare.symmetric_difference(t1, t2)
    except Exception:
        return None
