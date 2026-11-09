"""
GNN Sequence Corrector — leaf-pair cophenetic model
===================================================
The live model is `GraphSequenceCorrector` used with `topology_only=True,
pairwise_mrca=True`.  It takes a sequence of trees (one per genomic window) and
predicts, for each tree, the leaf-pair cophenetic matrix M[i,j] = size of the
smallest clade containing leaves i and j.  That matrix is permutation-invariant
over internal nodes and uniquely determines the topology, so it is well-posed
where per-internal-slot targets were not (see gnn_topology.py for the
rationale and the training loss).

  Backbone (shared):
      MPNN over the input tree  →  bidirectional GRU over the window sequence
      →  contextual node embeddings h ∈ R^{B,N,T,2H}

  Pairwise head (pairwise_mrca=True):
      learned projection of the LEAF embeddings  →  Euclidean distance in a
      learned space  →  predicted cophenetic matrix [B, T, S, S]  (symmetric,
      ≥0, metric-consistent).  Decode topology downstream by agglomerative
      clustering / neighbour-joining (see gnn_topology.py).

Node identity is a leaf/internal TYPE embedding (permutation-equivariant and
size-agnostic) by default; a legacy per-node IDENTITY embedding is available via
`node_id_dim > 0`.  The model is size-dynamic: leaf count S (and N = 2S-1) is
inferred per batch, so one model runs on any tree size.

Note: the class also contains an older parent-logit (Stage A) path, kept for
checkpoint compatibility but unused by the live pairwise-cophenetic
configuration.

Data assumptions
────────────────
- NUM_SAMPLES leaves first (indices 0..S-1), internal nodes after, higher
  index = older.  Root = node index NUM_NODES - 1 (by convention).
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Imports & constants
# ─────────────────────────────────────────────────────────────────────────────
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing

torch.manual_seed(42)

# ── Adjust to match your data ────────────────────────────────────────────────
NUM_NODES      = 11
NUM_SAMPLES    = 4
NUM_INTERNAL   = NUM_NODES - NUM_SAMPLES
ROOT_IDX       = NUM_NODES - 1

NODE_FEAT_DIM  = 2 + NUM_NODES            # genotype + timing + one-hot ID
EDGE_FEAT_DIM  = 1                        # branch length
SEQ_LEN        = 20
BATCH_SIZE     = 16


# ─────────────────────────────────────────────────────────────────────────────
# 1. MPNN layer  (MLP update — stable gradients)
# ─────────────────────────────────────────────────────────────────────────────
class MPNNLayer(MessagePassing):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__(aggr="add")
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.ReLU(),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([aggr_out, x], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Corrector
# ─────────────────────────────────────────────────────────────────────────────
class GraphSequenceCorrector(nn.Module):
    """
    Inputs : sequence of SEQ_LEN graphs (each [N nodes])
    Outputs: dict
        - pairwise_mrca=True (live) : {"mrca", "leaf_embed", "leaf_embed_metric"}
        - else (legacy Stage A)     : {"parent_logits": [B, T, N, N]}
    """
    def __init__(self,
                 node_feat_dim     : int = NODE_FEAT_DIM,
                 edge_feat_dim     : int = EDGE_FEAT_DIM,
                 node_hidden       : int = 64,
                 num_mp_layers     : int = 3,
                 gru_hidden        : int = 64,
                 num_gru_layers    : int = 1,
                 parent_hidden     : int = 32,
                 num_nodes         : int = NUM_NODES,
                 num_samples       : int = NUM_SAMPLES,
                 topology_only     : bool = False,
                 node_id_dim       : int = 0,
                 type_dim          : int = 0,
                 pairwise_mrca     : bool = False,
                 metric_dim        : int = 32,
                 jk_mode           : Optional[str] = None,
                 grad_checkpoint   : bool = False,
                 ):
        super().__init__()
        self.num_nodes     = num_nodes
        self.num_samples   = num_samples
        self.num_internal  = num_nodes - num_samples
        self.topology_only = topology_only
        self.pairwise_mrca = pairwise_mrca
        # Gradient checkpointing of the per-timestep MPNN encode: recompute the SEQ_LEN encode
        # activations in the backward pass instead of storing all of them, cutting the dominant
        # memory term (B·SEQ_LEN·edges·mp·hidden) at ~25-40% extra compute.  EXACT same gradients
        # (no dropout in the encoder), so accuracy is unchanged.  Only active in training.
        self.grad_checkpoint = grad_checkpoint
        # Inference-only speed knob: the per-timestep encode is launch-bound on GPU (thousands of
        # tiny B=1 graph forwards).  In eval mode we fuse a chunk of timesteps into ONE block-diagonal
        # encode, cutting kernel launches by ~encode_chunk×.  Graphs never share edges so batching does
        # not MIX them, but it is not byte-identical: the fused GEMMs reduce in a different order, so
        # outputs match the per-timestep loop only up to float reassociation (~2e-4 on the mrca; well
        # inside the model's own noise, and it flips a negligible fraction of decode ties — measured).
        # Chunk size bounds the transient encode activation memory (~chunk·edges·hidden under no_grad);
        # GNN_ENCODE_CHUNK overrides; set 1 for the exact byte-for-byte legacy path.  0/None → all T.
        self.encode_chunk = int(os.environ.get("GNN_ENCODE_CHUNK", "64")) or None
        # Jumping-knowledge: feed the head a combination of ALL MP layers' outputs (incl. the
        # pre-MP projection x^0) instead of only the last (over-smoothed) layer, so local
        # (few-hop) detail survives deep message passing.  None = off (last layer only).
        self.jk_mode = jk_mode if jk_mode in ("cat", "max", "sum") else None

        # ── Shared backbone ──
        # node_id_embed: per-node IDENTITY (fixed N, breaks permutation-equivariance) — legacy.
        # type_embed: per-node TYPE (leaf vs internal) — 2 vectors, permutation-equivariant
        #   and size-agnostic.  Leaf↔output correspondence is positional (row order), NOT from
        #   either embedding, so it is preserved regardless.
        self.node_id_embed   = nn.Embedding(num_nodes, node_id_dim) if node_id_dim > 0 else None
        self.type_embed      = nn.Embedding(2, type_dim) if type_dim > 0 else None
        if self.type_embed is not None:
            self.register_buffer("node_types",
                                 (torch.arange(num_nodes) < num_samples).long())  # 1=leaf, 0=internal
        self.node_input_proj = nn.Linear(node_feat_dim + node_id_dim + type_dim, node_hidden)
        self.mp_layers = nn.ModuleList([
            MPNNLayer(node_hidden, edge_feat_dim, node_hidden)
            for _ in range(num_mp_layers)
        ])
        # 'cat' JK concatenates x^0..x^L (L+1 vectors) then projects back to node_hidden;
        # 'max'/'sum' are parameter-free and dim-preserving.
        self.jk_proj = (nn.Linear((num_mp_layers + 1) * node_hidden, node_hidden)
                        if self.jk_mode == "cat" else None)
        # 'max'/'sum' leave no parameter in the state_dict, so flag the mode there for infer_and_load
        if self.jk_mode in ("max", "sum"):
            self.register_buffer(f"jk_{self.jk_mode}", torch.tensor(True))
        self.node_gru = nn.GRU(
            input_size    = node_hidden,
            hidden_size   = gru_hidden,
            num_layers    = num_gru_layers,
            dropout       = 0.1 if num_gru_layers > 1 else 0.0,
            batch_first   = True,
            bidirectional = True,
        )

        h2 = gru_hidden * 2

        # ── Stage A: parent prediction (pairwise scores) ──
        self.parent_q = nn.Linear(h2, parent_hidden)   # potential parent (older)
        self.parent_k = nn.Linear(h2, parent_hidden)   # potential child  (younger)

        # ── Pairwise leaf-MRCA head (cophenetic / clade-size) ──
        #   Predicts a symmetric S×S matrix of MRCA clade sizes purely from leaf
        #   embeddings.  Leaves have a stable identity along the sequence AND
        #   across input↔true, so this target is well-posed — unlike per-internal
        #   slot prediction, where the slot's meaning drifts.  The predicted
        #   value is a Euclidean distance in a learned space (symmetric, ≥0, and
        #   metric-consistent, which matches the ultrametric cophenetic matrix).
        if pairwise_mrca:
            self.metric_proj = nn.Sequential(
                nn.Linear(h2, h2), nn.ReLU(), nn.Linear(h2, metric_dim),
            )
            self.metric_scale = nn.Parameter(torch.tensor(1.0))
            self.metric_bias  = nn.Parameter(torch.tensor(0.0))

        # Precompute mask: a parent must be strictly older (higher index) than
        # its child.  Forbid i <= j (upper triangle incl. diagonal), but allow
        # the root self-loop so its softmax column is well-defined — the root
        # has no parent and points to itself by convention.
        parent_mask = torch.triu(
            torch.ones(num_nodes, num_nodes, dtype=torch.bool), diagonal=0)
        parent_mask[num_nodes - 1, num_nodes - 1] = False
        self.register_buffer("parent_mask", parent_mask)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Accept a max/sum checkpoint saved before the jk flag existed (the caller chose the mode)."""
        if self.jk_mode in ("max", "sum"):
            state_dict = {f"jk_{self.jk_mode}": torch.tensor(True), **state_dict}
        return super().load_state_dict(state_dict, *args, **kwargs)

    # ── Encode one time step using INPUT topology ──
    def _encode_impl(self, x_in, edge_index, edge_attr, B, N, S):
        """Tensor-only body (so it can be wrapped by torch.utils.checkpoint)."""
        feats = [x_in]
        if self.node_id_embed is not None:        # legacy per-node identity — size-TIED
            ids = torch.arange(N, device=x_in.device).repeat(B)
            feats.append(self.node_id_embed(ids))
        if self.type_embed is not None:           # leaf iff within-graph index < S — size-agnostic
            tids = (torch.arange(N, device=x_in.device) < S).long().repeat(B)
            feats.append(self.type_embed(tids))
        x = self.node_input_proj(torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0])
        if self.jk_mode is None:                                  # last-layer only (default)
            for layer in self.mp_layers:
                x = F.relu(layer(x, edge_index, edge_attr))
            return x                                              # [B*N, node_hidden]
        # Jumping-knowledge: keep every layer's output (x^0 = pre-MP, most local) and combine
        outs = [x]
        for layer in self.mp_layers:
            x = F.relu(layer(x, edge_index, edge_attr))
            outs.append(x)
        if self.jk_mode == "cat":
            return self.jk_proj(torch.cat(outs, dim=-1))          # [B*N, node_hidden]
        stacked = torch.stack(outs, dim=0)                        # [L+1, B*N, node_hidden]
        return stacked.max(dim=0).values if self.jk_mode == "max" else stacked.sum(dim=0)

    def encode_graphs(self, batch: Batch) -> torch.Tensor:
        # Infer nodes-per-graph from the batch (uniform within a batch) so the model
        # is size-dynamic: a type-embedding model runs on any leaf count S (N = 2S-1).
        B = batch.num_graphs
        N = batch.num_nodes // B
        S = (N + 1) // 2
        if self.grad_checkpoint and self.training:
            # use_reentrant=False handles non-grad inputs (x/edge_index) + module params correctly.
            return checkpoint(self._encode_impl, batch.x, batch.edge_index, batch.edge_attr,
                              B, N, S, use_reentrant=False)
        return self._encode_impl(batch.x, batch.edge_index, batch.edge_attr, B, N, S)

    def _encode_sequence_batched(self, sequences, B, N, S, chunk):
        """Encode the whole timestep sequence, fusing `chunk` timesteps per forward.  Each timestep
        is a Batch of B graphs (B·N nodes); a chunk of C of them is the block-diagonal graph on
        C·B·N nodes — each timestep's edges shifted by its slot's node offset so no edges cross.  MP
        never mixes disconnected components, so this returns EXACTLY the per-timestep encodes, only
        with C×-fewer kernel launches.  Returns a list of T tensors [B·N, H], matching encode_graphs.
        Only used in eval mode (the grad_checkpoint training path keeps its per-timestep granularity)."""
        BN = B * N
        out = []
        T = len(sequences)
        step = chunk if chunk else T
        for t0 in range(0, T, step):
            grp = sequences[t0:t0 + step]
            C = len(grp)
            x = torch.cat([g.x for g in grp], dim=0)                          # [C·B·N, F]
            ei = torch.cat([g.edge_index + c * BN for c, g in enumerate(grp)], dim=1)
            ea = torch.cat([g.edge_attr for g in grp], dim=0)
            h = self._encode_impl(x, ei, ea, C * B, N, S)                     # [C·B·N, H]
            out.extend(h[c * BN:(c + 1) * BN] for c in range(C))
        return out

    def forward(self, sequences: list) -> dict:
        B = sequences[0].num_graphs
        T = len(sequences)
        N = sequences[0].num_nodes // B          # nodes-per-graph, inferred (size-dynamic)
        S = (N + 1) // 2                          # leaves, inferred (used by the batched encode)

        # 1. Encode each time step with the input topology.  Training (and grad_checkpoint) keeps the
        #    per-timestep loop; inference fuses chunks of timesteps into one block-diagonal encode.
        if self.training or self.grad_checkpoint:
            per_step = [self.encode_graphs(b) for b in sequences]
        else:
            per_step = self._encode_sequence_batched(sequences, B, N, S, self.encode_chunk)
        per_step = [e.view(B, N, -1) for e in per_step]          # list of [B, N, H]
        node_embeds = torch.stack(per_step, dim=2)               # [B, N, T, H]

        # 2. Bidirectional GRU over time (per node)
        gru_in = node_embeds.view(B * N, T, -1)
        all_h, _ = self.node_gru(gru_in)                         # [B*N, T, 2H]
        all_h = all_h.view(B, N, T, -1)                          # [B, N, T, 2H]

        # 2b. Pairwise leaf-MRCA head — predict cophenetic (clade-size) matrix
        if self.pairwise_mrca:
            S = (N + 1) // 2                                      # leaves, inferred (size-dynamic)
            L = all_h[:, :S]                                      # [B, S, T, H]
            P = self.metric_proj(L).permute(0, 2, 1, 3)          # [B, T, S, D]
            sq   = (P * P).sum(-1)                                # [B, T, S]
            gram = torch.matmul(P, P.transpose(-1, -2))          # [B, T, S, S]
            d2   = (sq.unsqueeze(-1) + sq.unsqueeze(-2)
                    - 2.0 * gram).clamp_min(1e-9)                # [B, T, S, S]
            mrca = self.metric_scale * d2.sqrt() + self.metric_bias
            # Exposed for probing (Step 2); harmless for training, which only reads "mrca".
            #   leaf_embed        = L : contextual GRU leaf embeddings [B, T, S, H] (pre-metric, richer/raw)
            #   leaf_embed_metric = P : the metric space [B, T, S, D] whose pairwise Euclidean distances
            #                           ARE the tree (Euclidean-calibrated — better for PCA/nearest-neighbour)
            return {"mrca": mrca,
                    "leaf_embed": L.permute(0, 2, 1, 3),
                    "leaf_embed_metric": P}

        # 3. Stage A — parent scoring
        q = self.parent_q(all_h)                                 # [B, N, T, P]
        k = self.parent_k(all_h)                                 # [B, N, T, P]
        # parent_logits[b, t, i, j] = q[b, i, t] · k[b, j, t]
        parent_logits = torch.einsum('bith,bjth->btij', q, k)    # [B, T, N, N]

        # mask invalid parents: a parent must be strictly older (higher index)
        parent_mask = self.parent_mask.view(1, 1, N, N)          # [1, 1, N, N]
        parent_logits = parent_logits.masked_fill(parent_mask, float("-inf"))

        # Legacy Stage-A output: the parent-logit topology (unused by the live
        # pairwise-cophenetic head, which returns above).
        return {"parent_logits": parent_logits}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Data utilities
# ─────────────────────────────────────────────────────────────────────────────
def build_parent_index_from_topology(edge_index: torch.Tensor,
                                     num_nodes: int = NUM_NODES,
                                     root_idx:  int = ROOT_IDX) -> torch.Tensor:
    """
    Convert an undirected edge_index into parent_index [N].

    Convention: higher node index = older = parent.
    Adapt if your tree-building convention is different.
    """
    parent_index = torch.full((num_nodes,), -1, dtype=torch.long)
    half = edge_index.shape[1] // 2
    for s, d in zip(edge_index[0, :half].tolist(),
                    edge_index[1, :half].tolist()):
        p, c = (s, d) if s > d else (d, s)
        parent_index[c] = p
    return parent_index


# ── Denormal (subnormal) weight hygiene ─────────────────────────────────────
# Weight decay shrinks unused weights geometrically; with no underflow guard they slide BELOW the
# smallest normal float32 magnitude and park in the subnormal range instead of reaching 0.  They are
# numerically dead (a 1e-40 weight times an O(1) activation contributes ~1e-40 to a sum whose other
# terms are O(0.1) — below the representable precision of the result, so it cannot change anything),
# but x86 CPUs abort the vectorised fast path and trap to a microcode assist on subnormal operands.
# Measured on gnn_v7_varsize_rk0.5L3_knn7w_best_truesplit.pt (21.6% of params subnormal): 19x slower
# forward at S=100/knn7, with a BIT-IDENTICAL predicted cophenetic matrix (max abs diff 0.0).
# GPUs flush subnormals in hardware, so this is a CPU-inference fix only and training is unaffected.
#
# The threshold is EXACTLY the float32 normal/subnormal boundary and the safety argument depends on
# that: below it, values already carry degraded precision.  Do NOT raise it "to be safe" — a larger
# threshold zeroes live weights and WOULD change results.
DENORMAL_MIN = 1.1754944e-38          # smallest normal float32 magnitude (2^-126)


def flush_denormals(obj, verbose=True):
    """Zero every subnormal float in a module or a state_dict, IN PLACE; returns the count.

    `obj` may be an nn.Module (parameters + buffers) or a state_dict.  Non-float tensors are skipped
    (e.g. the bool `parent_mask` buffer, where an abs() comparison would be meaningless).  The count
    is reported because it is diagnostic, not incidental: a large fraction means weight decay has
    nulled that share of the model, which is worth knowing independently of the speed-up."""
    if isinstance(obj, dict):
        tensors = list(obj.values())
    else:
        tensors = list(obj.parameters()) + list(obj.buffers())
    n_sub = n_tot = 0
    with torch.no_grad():
        for t in tensors:
            if not torch.is_tensor(t) or not t.is_floating_point():
                continue
            mask = (t != 0) & (t.abs() < DENORMAL_MIN)   # exclude exact zeros so the count is honest
            c = int(mask.sum())
            if c:
                t[mask] = 0.0
            n_sub += c
            n_tot += t.numel()
    if verbose and n_sub:
        print(f"  flushed {n_sub}/{n_tot} subnormal params ({100 * n_sub / n_tot:.2f}%) → 0 "
              f"(output unchanged; CPU inference no longer traps on them)")
    return n_sub


def save_flushed(model, path, verbose=True):
    """torch.save the model's weights with subnormals zeroed, WITHOUT touching the live model.

    The clone matters: state_dict() returns references to the real parameters, so flushing it in
    place would mutate the model mid-training and desynchronise it from the optimiser's moment
    buffers.  Cloning keeps this purely an artifact-level fix."""
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    flush_denormals(sd, verbose=verbose)
    torch.save(sd, path)


def infer_and_load(path, map_location="cpu", return_cfg=False, verbose=True):
    """Rebuild a GraphSequenceCorrector from a checkpoint by INFERRING its architecture from
    the saved state_dict — hidden width, MP depth, GRU hidden/#layers, metric dim, type/id
    embedding dims, node-feature width, jumping-knowledge, and num_nodes — then load the weights
    and return the eval-mode model.  So callers only need the checkpoint path, not the training
    hyperparameters.  With return_cfg=True, also returns the inferred config dict (handy to check
    the node-feature width and max leaf count (num_nodes+1)//2 before feeding data)."""
    sd = torch.load(path, map_location=map_location)
    nh = sd["node_input_proj.weight"].shape[0]
    in_dim = sd["node_input_proj.weight"].shape[1]
    type_dim = sd["type_embed.weight"].shape[1] if "type_embed.weight" in sd else 0
    node_id_dim = sd["node_id_embed.weight"].shape[1] if "node_id_embed.weight" in sd else 0
    feat = in_dim - type_dim - node_id_dim
    edge_feat = sd["mp_layers.0.message_mlp.0.weight"].shape[1] - 2 * nh
    n_mp = len({k.split(".")[1] for k in sd if k.startswith("mp_layers.")})
    gru_hidden = sd["node_gru.weight_ih_l0"].shape[0] // 3
    n_gru = len({k.split("_l")[1].replace("_reverse", "") for k in sd if "weight_ih_l" in k})
    metric_dim = sd["metric_proj.2.weight"].shape[0]
    num_nodes = sd["node_types"].shape[0] if "node_types" in sd else sd["parent_mask"].shape[0]
    jk = ("cat" if "jk_proj.weight" in sd else "max" if "jk_max" in sd else "sum" if "jk_sum" in sd
          else (os.environ.get("GNN_JK", "").lower() or None))
    cfg = dict(node_feat_dim=feat, edge_feat_dim=edge_feat, node_hidden=nh, num_mp_layers=n_mp,
               gru_hidden=gru_hidden, num_gru_layers=n_gru, metric_dim=metric_dim, type_dim=type_dim,
               node_id_dim=node_id_dim, num_nodes=num_nodes, num_samples=(num_nodes + 1) // 2,
               jk_mode=jk, pairwise_mrca=True, topology_only=True)
    if verbose:
        print(f"  inferred arch: hidden={nh} mp={n_mp} gru={gru_hidden}x{n_gru} metric={metric_dim} "
              f"type={type_dim} feat={feat} edge_feat={edge_feat} jk={jk} N={num_nodes}")
    m = GraphSequenceCorrector(**cfg)
    m.load_state_dict(sd)
    m.eval()
    flush_denormals(m, verbose=verbose)          # defensive: fixes checkpoints saved before save_flushed
    return (m, cfg) if return_cfg else m
