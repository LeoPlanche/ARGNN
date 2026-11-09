"""
branch_length.py — branch-length / node-TIME estimation, SEPARATE from topology inference.
=============================================================================================
Topology is solved elsewhere (true sims; Relate/tsinfer/SINGER; or our model's avg_constr decode).
This module ONLY estimates *times* on a given topology, and scores dating accuracy vs the truth.

Two stages, mirroring how Relate/tsinfer work (topology first, then dating):
  (1) date_ts(ts, method=...)         — estimate node times on a fixed topology+mutations.
        method="tsdate"  : the Relate/tsinfer-family approach (tskit-native Bayesian dater).
        method="dl"      : PLACEHOLDER for the future deep-learning dater (same in/out contract).
  (2) dating_accuracy(inf_ts, true_ts) — pairwise log-TMRCA correlation + log-RMSE vs truth
        (leaf-pair based, so it's robust to node-id / topology differences between the two).

Adapter (so we can date OUR topology):
  model_decode_to_ts(ckpt, true_trees) — assemble the model's per-position avg_constr decode into a
        tskit tree sequence with each SNP's mutation placed on its carrier-clade branch (avg_constr
        guarantees the carriers ARE a clade → one branch), ready for date_ts.  v1 uses global
        clade→node identity (max mutation pooling); node-sharing scheme is the main design knob.

The model contract is identical for tsdate and the future DL dater: (topology + mutations) → node
times.  So a DL method drops into date_ts(method="dl") and is judged by the same dating_accuracy.

Run: sweeps scenarios×sizes×files and aggregates per (scenario,size), like the topology eval.
    GNN_VS_DIR=simulated_varsize GNN_CKPT=gnn_v7_varsize_best.pt \
    conda run --no-capture-output -n lplancheenv python -u branch_length.py
Env: GNN_VS_SCENARIOS (default const,split,bott), GNN_BL_SIZES (default = train+test sizes),
     GNN_BL_FILES (default 3; files taken from the held-out tail of the range), GNN_CKPT (set → also date
     our model's topology), GNN_BL_ASSEMBLY (contiguous|global|bridge|bridge_noroot), GNN_BL_EPS (bridge threshold),
     GNN_BL_BREAKPOINTS (next_snp|midpoint, where a tree changes between two SNPs),
     GNN_BL_REFINE_AGE (opt-in age-consistency 2nd pass), GNN_BL_AGE_TOL, GNN_DEVICE ("" auto / cuda / cpu),
     GNN_DATING_METHOD (tsdate|polegon|dl; dater for the model+true rows), GNN_POLEGON_DIR/_N/_THIN/_SCALING/_BURN_IN,
     GNN_BL_RELATE_FILE (relate_vc default = variable-Ne Relate; 'relate' = constant-Ne),
     GNN_DATING_NE, GNN_MUT_RATE (must match the simulation — 1.25e-8 for the new data!), GNN_VS_DIR.
Needs: pip install tsdate  (tskit-native; compatible with tskit 1.0.x).
"""
import os
import time
import multiprocessing as mp
from collections import defaultdict
from math import comb
import numpy as np
import tskit
try:
    import tsdate
except Exception:                               # optional until installed
    tsdate = None

import train as V
from train import _flat_stream, _sliding_predict, decode_model, HAP_K

MUT_RATE = float(os.environ.get("GNN_MUT_RATE", "1.25e-8"))
NE       = int(os.environ.get("GNN_DATING_NE", "10000"))
DATA_DIR = V.DATA_DIR
# Runner knobs (GNN_* env vars, matching the rest of the codebase). The runner SWEEPS scenarios×sizes
# ×files and aggregates per (scenario,size) — like the topology eval — not a single file.
BL_SCENARIOS = [s for s in os.environ.get("GNN_VS_SCENARIOS", ",".join(V.SCENARIOS)).split(",") if s]
BL_SIZES     = [int(x) for x in os.environ.get("GNN_BL_SIZES",
                ",".join(map(str, V.TRAIN_SIZES + V.TEST_SIZES))).split(",")]
BL_FILES     = int(os.environ.get("GNN_BL_FILES", "3"))  # # files per (scenario,size); taken from the
                                                         # held-out tail of the file range (leakage-free)
DATING_METHOD = os.environ.get("GNN_DATING_METHOD", "tsdate")
# Which Relate trees back the "relate" method: 'relate_vc' (default) = Relate's variable-Ne
# EstimatePopulationSize re-dating (its RECOMMENDED pipeline, the fair best-practice baseline — it
# removes the constant-Ne under-dating bias on structured demographies like split); 'relate' = the
# constant-Ne version.  Falls back to _relate.trees if _relate_vc.trees is absent.  Column stays "relate".
RELATE_FILE = os.environ.get("GNN_BL_RELATE_FILE", "relate_vc")
CKPT        = os.environ.get("GNN_CKPT", "")             # set → also date OUR model's topology
BL_ASSEMBLY = os.environ.get("GNN_BL_ASSEMBLY", "bridge")  # model→ts node-sharing: contiguous|global|bridge|
                                                        # bridge_noroot  (A/B for dating: contiguous=unbiased/
                                                        #  noisier, global=more pooling/lower-variance but
                                                        #  conflates, bridge=size-aware middle (see GNN_BL_EPS),
                                                        #  bridge_noroot=bridge but the ROOT is keyed on its top
                                                        #  split not the trivial full-set clade → distinct root
                                                        #  nodes/ages per recombination, fixes the time ceiling)
EPS         = float(os.environ.get("GNN_BL_EPS", "1.0"))  # bridge: tolerated expected coincidental merges;
                                                        # bridge a clade across a gap iff (#intervening trees)
                                                        # · p_k < EPS. Higher → more bridging (→ global).
ROOT_TOL    = int(os.environ.get("GNN_BL_ROOT_TOL", "0"))  # bridge_noroot ONLY: keep the same root node while the
                                                        # top split stays within this many SWITCHES (unit set by
                                                        # ROOT_METRIC) of the reference split (0 = exact match).
                                                        # Pools mutations across near-identical roots (ages ~equal)
                                                        # → lower dating variance, low bias.
ROOT_METRIC = os.environ.get("GNN_BL_ROOT_METRIC", "leaf")  # ROOT_TOL unit: 'leaf' = #leaves that switch sides;
                                                        # 'clade' = #CLADE switches (≈ recombination events — a
                                                        # whole clade crossing the root = 1, since one SPR moves a
                                                        # subtree intact). 'clade' returns 0/1/∞ (TOL≥1 ⇒ allow a
                                                        # single clade switch); superset of 'leaf' TOL=1 (a lone
                                                        # leaf IS a clade). Meaningful with ROOT_CMP=adjacent.
ROOT_CMP    = os.environ.get("GNN_BL_ROOT_CMP", "anchor")  # bridge_noroot ROOT_TOL reference: 'anchor' = the run's
                                                        # FIRST split (bounds total drift, can't re-collapse);
                                                        # 'adjacent' = the PREVIOUS split (tracks per-recombination
                                                        # change, but a slow monotonic drift can chain into one
                                                        # big root node). Identical at ROOT_TOL=0.
REFINE_AGE  = os.environ.get("GNN_BL_REFINE_AGE", "0") == "1"  # OPT-IN 2nd pass: assemble contiguous → date →
                                                        # re-bridge requiring the two ages to AGREE (joint
                                                        # node-identity+time, à la SINGER). Only with bridge.
AGE_TOL     = float(os.environ.get("GNN_BL_AGE_TOL", "0.5"))  # max |Δ log-age| to bridge across a gap (refine)
BREAKPOINTS = os.environ.get("GNN_BL_BREAKPOINTS", "next_snp")  # where the tree changes between the SNPs x and y:
                                                        # 'next_snp' = at y; 'midpoint' = at x + ceil((y - x) / 2),
                                                        # the midpoint rounded up to a whole number, which lies
                                                        # after x and at or before y for whole-number positions.
                                                        # Only the breakpoints move: the nodes and the tree and
                                                        # mutations at every SNP are the same.
DEVICE      = os.environ.get("GNN_DEVICE", "")          # "", "cuda", "cuda:0", "cpu"; "" → auto
DECODE_WORKERS = int(os.environ.get("GNN_DECODE_WORKERS", "1"))  # per-position decode is embarrassingly parallel
                                                        # (each site's constrained-UPGMA is independent). 1 = serial
                                                        # (bit-identical legacy path); >1 fans the S³ decode out over
                                                        # that many fork workers; 0/<0 = use all CPUs (os.cpu_count()).
                                                        # Assembly + tsdate stay serial; this only speeds the decode
                                                        # stage (the 43%-of-wall slice on the server benchmark).
SQUASH      = os.environ.get("GNN_BL_SQUASH", "1") == "1"  # 1 → edge-squash the assembled model ts before dating.
                                                        # Our per-position assembly emits one edge record per (tree,
                                                        # branch) with almost no span-merging (~9× more edges than
                                                        # tsinfer). simplify(keep_unary=True) merges a persistent
                                                        # (parent,child) over adjacent intervals into one wide-span
                                                        # edge — identical topology/nodes/sites/mutations, far fewer
                                                        # edges. Since tsdate cost ∝ edge count, this cuts dating ~6×
                                                        # with NO accuracy change (verified vs truth: pearson/rmse/bias
                                                        # match within noise under the bridge_noroot config).
# Per-PAIR TMRCA table + CSV (OPT-IN via GNN_BL_NODECSV): one row = one leaf pair at a position, with the
# TRUE tmrca and every method's tmrca for that pair.  Pairwise ⇒ every method always has a value (no node
# matching, no NAs) and it's exactly the quantity dating_accuracy already uses.  Sampling is a plain
# UNIFORM random subset of all pairs, so the CSV reflects the NATURAL pairwise-TMRCA distribution
# (deep pairs — the MRCA of ~S²/4 pairs — dominate, as they should); reweight downstream if wanted.
NODE_CSV    = os.environ.get("GNN_BL_NODECSV", "")                 # CSV path; "" = feature off
NODE_MAX    = int(os.environ.get("GNN_BL_MAXNODES", "50000"))      # cap total pair rows per run (spread over cells)
NODE_STRIDE = int(os.environ.get("GNN_BL_NODE_STRIDE", "20"))      # position subsample for pair collection
_NODE_TAGS  = ("true", "relate", "tsinfer", "singer", "model")     # method columns (true = dater on true topology)
# POLEGON: a PRIOR-OBLIVIOUS branch-length re-calibrator (MCMC on a fixed topology with branch-mapped
# mutations) — an alternative dater to tsdate (select with GNN_DATING_METHOD=polegon). No coalescent prior,
# which is appealing given tsdate's bottleneck bias.  Needs a tskit .trees with mutations on branches and
# LINKED trees (adjacent trees differ little) — our assembled ts satisfies both.
POLEGON_DIR  = os.environ.get("GNN_POLEGON_DIR",
                              os.path.join(os.path.dirname(os.path.abspath(__file__)), "POLEGON/POLEGON/POLEGON"))
POLEGON_MASTER = os.path.join(POLEGON_DIR, "polegon_master")
POLEGON_N    = int(os.environ.get("GNN_POLEGON_N", "100"))       # # posterior MCMC samples (averaged)
POLEGON_THIN = int(os.environ.get("GNN_POLEGON_THIN", "10"))     # MCMC thinning interval
POLEGON_SCALING = int(os.environ.get("GNN_POLEGON_SCALING", "5"))  # # rescaling steps after MCMC
POLEGON_BURN_IN = int(os.environ.get("GNN_POLEGON_BURN_IN", "100"))  # MCMC burn-in (-burn_in)


def _resolve_device():
    import torch
    if DEVICE:
        return DEVICE
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── (1) dating methods (pluggable) ──────────────────────────────────────────────
def date_ts(ts, method="tsdate", mutation_rate=MUT_RATE, Ne=NE, **kw):
    """Estimate node times on a FIXED topology (ts must carry mutations).  Returns a dated ts.
    method='tsdate' = the Relate/tsinfer-family dater; 'polegon' = prior-oblivious MCMC re-calibrator;
    'dl' reserved for the learned dater."""
    if method == "tsdate":
        if tsdate is None:
            raise ImportError("tsdate not installed — `pip install tsdate`")
        # tsdate re-estimates node times from per-branch mutation counts; existing times are ignored.
        # The default method (variational_gamma) is mutation-rate-driven and does NOT take a fixed Ne
        # (it infers the prior), so we pass only mutation_rate.  Ne is forwarded only if the caller
        # explicitly selects a population-size-based method via **kw (e.g. method="inside_outside").
        if kw.get("method", "variational_gamma") == "variational_gamma":
            # tsdate's rescaling step splits the time axis into `rescaling_intervals` (default 1000)
            # mutation-density epochs and rescales each; on a sparse / low-mutation ts that's too many
            # → the epoch breakpoints collapse ("AssertionError: Use fewer rescaling intervals").  Step
            # DOWN the interval count (keeping a coarser version of the correction) and only DISABLE it
            # (0) as the last resort, rather than dropping the whole file.
            attempts = [kw] if "rescaling_intervals" in kw else \
                       [kw] + [dict(kw, rescaling_intervals=ri) for ri in (100, 10, 0)]
            for i, k in enumerate(attempts):
                try:
                    # match_segregating_sites=True, as run_external_tools.date_with_tsdate dates ARGNN in the
                    # benchmark.
                    return tsdate.date(ts, mutation_rate=mutation_rate, match_segregating_sites=True, **k)
                except AssertionError:
                    if i == len(attempts) - 1:
                        # Not the rescaling-intervals issue: the variational EP itself hit a numerical
                        # assertion (e.g. `assert 0 < x[1]` in variational._rescale — a gamma rate went
                        # ≤0 under extreme branch-length ratios, as with deep archaic + outgroup trees).
                        # Fall back to the older DISCRETIZED dater (inside_outside), which doesn't use the
                        # EP path and preserves the topology; slower but robust.  population_size REQUIRED.
                        print("    [tsdate variational_gamma failed (EP numerical assertion); "
                              "falling back to method='inside_outside']", flush=True)
                        return tsdate.date(ts, mutation_rate=mutation_rate,
                                           method="inside_outside", population_size=Ne)
        return tsdate.date(ts, mutation_rate=mutation_rate, population_size=Ne, **kw)
    if method == "polegon":
        return _polegon_date(ts, mutation_rate)
    if method == "dl":
        raise NotImplementedError("deep-learning dater: future work (same (topology,muts)->times contract)")
    raise ValueError(f"unknown dating method {method!r}")


def _polegon_date(ts, mutation_rate):
    """Re-calibrate branch lengths with POLEGON (prior-oblivious MCMC).  Dumps the input ts, runs
    polegon_master (with THIS interpreter, so it finds the env's tskit/pandas), loads the posterior-average
    output.  ts must have branch-mapped mutations and linked trees."""
    import tempfile, subprocess, sys, shutil
    if not os.path.exists(POLEGON_MASTER):
        raise ImportError(f"polegon_master not found at {POLEGON_MASTER} (set GNN_POLEGON_DIR)")
    tmp = tempfile.mkdtemp(prefix="polegon_")
    try:
        inp, out = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        ts.dump(inp + ".trees")
        cmd = [sys.executable, POLEGON_MASTER, "-m", str(mutation_rate),
               "-input", inp, "-output", out, "-burn_in", str(POLEGON_BURN_IN),
               "-num_samples", str(POLEGON_N), "-thin", str(POLEGON_THIN),
               "-scaling_rep", str(POLEGON_SCALING)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        ages_file = inp + "_new_nodes.txt"
        if not os.path.exists(ages_file):
            raise RuntimeError(f"polegon produced no node ages (rc={r.returncode}): "
                               f"{(r.stderr or r.stdout)[-400:]}")
        # POLEGON's node ages on the input ts, as singer-snakemake does: polegon_master's own rebuild
        # sets no mutation parents, which tskit 1.0 rejects at a site with nested mutations
        tables = ts.dump_tables()
        tables.nodes.time = np.loadtxt(ages_file, ndmin=1)
        tables.mutations.time = np.full(tables.mutations.num_rows, tskit.UNKNOWN_TIME)
        tables.sort()
        return tables.tree_sequence()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── (2) dating-accuracy metric — pairwise TMRCA in TIME, robust to topology/node-id diffs ────────
def tmrca_matrix(tree, S):
    """[S,S] time of the MRCA (generations) for every leaf pair; diagonal 0."""
    M = np.zeros((S, S), dtype=float)
    for a in range(S):
        for b in range(a + 1, S):
            M[a, b] = M[b, a] = tree.time(tree.mrca(a, b))
    return M


def dating_accuracy(inf_ts, true_ts, stride=8, max_pos=2000, eps=1.0, gpool=None):
    """Compare inferred vs TRUE coalescence times at shared SNP positions, via the pairwise log-TMRCA
    vectors (one number per leaf pair).  Returns RAW per-position lists so callers can POOL across
    files: dict(pearson_log=[...], spearman=[...], rmse_log=[...], bias_log=[...]).  Leaf-pair based ⇒
    needs no node matching and tolerates topology differences.  Use summarize() for the means.
    If `gpool` is given, every pair (not just the per-position summary) is streamed into it for the
    GLOBAL-POOLED pairwise block (see _summarize_g)."""
    S = true_ts.num_samples
    iu = np.triu_indices(S, 1)
    out = {k: [] for k in ("pearson_log", "spearman", "rmse_log", "bias_log")}
    positions = [s.position for s in true_ts.sites()]
    n = 0
    for p in positions[::stride]:
        try:
            ti = np.log(tmrca_matrix(inf_ts.at(p), S)[iu] + eps)
            tt = np.log(tmrca_matrix(true_ts.at(p), S)[iu] + eps)
        except Exception:
            continue
        if ti.std() < 1e-9 or tt.std() < 1e-9:
            continue
        out["pearson_log"].append(float(np.corrcoef(ti, tt)[0, 1]))
        out["spearman"].append(_spearman(ti, tt))
        out["rmse_log"].append(float(np.sqrt(np.mean((ti - tt) ** 2))))
        out["bias_log"].append(float(np.mean(ti - tt)))
        if gpool is not None:
            _gpool_add(gpool, ti, tt)
        n += 1
        if n >= max_pos:
            break
    return out


def summarize(acc):
    """Mean of each raw metric list (+ n)."""
    m = lambda x: float(np.mean(x)) if x else float("nan")
    return {k: m(v) for k, v in acc.items()} | {"n": len(acc["pearson_log"])}


# ── GLOBAL-POOLED pairwise error ──────────────────────────────────────────────────────────────────
# summarize()/dating_accuracy() average the per-POSITION rmse; that sqrt-compresses within a position
# and dilutes the rare local trees a method gets badly wrong.  The block below instead POOLS every
# leaf pair (over positions+files) into one distribution, so the L2 rmse_log is dominated by the
# catastrophic-tail pairs — the honest end-to-end pairwise error.  It also reports median|log-err|
# (typical accuracy) and p99 (tail), so "modestly worse on average but occasionally catastrophic"
# (e.g. the model) is visible, not hidden.  Running sums = no storage; a per-position subsample of
# |err| (GRES_PER_POS) feeds the percentiles.
def _new_gpool():
    return {"n": 0, "se": 0.0, "se2": 0.0, "sx": 0.0, "sy": 0.0,
            "sxx": 0.0, "syy": 0.0, "sxy": 0.0, "sp": []}           # sp = per-position spearman

def _gpool_add(g, y_inf, x_true):
    """Accumulate one position's log-TMRCA vectors (y=inferred, x=true) into g.  Exact running sums
    give pearson/rmse/bias POOLED over all pairs; spearman is the per-position value (averaged later,
    same as summarize())."""
    e = y_inf - x_true
    g["n"]  += e.size
    g["se"] += float(e.sum());          g["se2"] += float((e * e).sum())
    g["sx"] += float(x_true.sum());     g["sy"]  += float(y_inf.sum())
    g["sxx"]+= float((x_true * x_true).sum())
    g["syy"]+= float((y_inf * y_inf).sum())
    g["sxy"]+= float((y_inf * x_true).sum())
    g["sp"].append(_spearman(y_inf, x_true))

def _gpool_merge(dst, src):
    for key in ("n", "se", "se2", "sx", "sy", "sxx", "syy", "sxy"):
        dst[key] += src[key]
    dst["sp"].extend(src["sp"])

def _summarize_g(g):
    n = g["n"]
    if not n:
        return {"pearson_log": float("nan"), "spearman": float("nan"), "rmse_log": float("nan"),
                "bias_log": float("nan"), "n": 0}
    cov = n * g["sxy"] - g["sx"] * g["sy"]
    vx  = n * g["sxx"] - g["sx"] ** 2
    vy  = n * g["syy"] - g["sy"] ** 2
    pear = cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")
    return {"pearson_log": float(pear), "spearman": float(np.mean(g["sp"])) if g["sp"] else float("nan"),
            "rmse_log": float(np.sqrt(g["se2"] / n)), "bias_log": float(g["se"] / n), "n": n}


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# ── adapter: model avg_constr decode → datable tree sequence ─────────────────────
def _clades_and_edges(parent, S):
    """From a parent-index tree → (list of (parent_cladeset, child_cladeset) edges, dict node→cladeset)."""
    N = len(parent)
    children = [[] for _ in range(N)]
    root = -1
    for i, p in enumerate(parent):
        p = int(p)
        (children[p].append(i) if p >= 0 else None)
        if p < 0:
            root = i
    clade = {}
    order, stack = [], [root]
    while stack:
        u = stack.pop(); order.append(u)
        for c in children[u]:
            stack.append(c)
    for u in reversed(order):
        clade[u] = (frozenset([u]) if u < S else frozenset().union(*[clade[c] for c in children[u]]))
    edges = [(clade[int(parent[c])], clade[c]) for c in range(N) if int(parent[c]) >= 0]
    return edges, clade


def _bipart_dist(ca, cb):
    """Leaf-switch distance between two root top-splits (each a set of child clades): the min number of
    leaves that must cross the split to turn one into the other.  Binary roots only; otherwise exact
    (0 if identical, else huge).  A single leaf moving sides → 1."""
    if ca == cb:
        return 0
    if len(ca) != 2 or len(cb) != 2:
        return 1 << 30
    A = next(iter(ca)); X, Y = tuple(cb)
    return min(len(A ^ X), len(A ^ Y))


def _clade_switch_dist(sp_a, cl_a, sp_b, cl_b):
    """Clade-switch (≈ recombination) distance between two root top-splits: 0 if identical, 1 if a single
    clade crossed the root, else huge.  sp = the root's 2 child clades; cl = the set of ALL clades of that
    tree.  One SPR prunes+regrafts a subtree intact, so a single crossing = a set of leaves that (i) moved
    in ONE direction and (ii) is a clade in BOTH trees (a subtree in the source, landed intact in the
    dest).  Binary roots only; bidirectional / non-clade / polytomy → huge (≥2 events, kept separate)."""
    if sp_a == sp_b:
        return 0
    if len(sp_a) != 2 or len(sp_b) != 2:
        return 1 << 30
    La, Ra = tuple(sp_a)
    for Lb, Rb in (tuple(sp_b), tuple(sp_b)[::-1]):                 # both pairings: at most one is one-way
        b, c = La & Rb, Ra & Lb                                      # leaves crossing each direction
        X = b or c                                                   # the single moved leaf-set
        if not (b and c) and X in cl_a and X in cl_b:
            return 1
    return 1 << 30


def _merge_intervals(intervals):
    iv = sorted(intervals); m = [list(iv[0])]
    for l, r in iv[1:]:
        if l <= m[-1][1] + 1e-6:
            m[-1][1] = max(m[-1][1], r)
        else:
            m.append([l, r])
    return m


def _age_at(lookup, cl, pos):
    """Age of clade `cl`'s MRCA at genomic position `pos`, from a pass-1 dated lookup (or None)."""
    for l, r, age in lookup.get(cl, ()):
        if l <= pos < r:
            return age
    return None


def _build_age_lookup(node_clade1, dated1):
    """clade → sorted [(left, right, age)] from a (contiguous-assembled) dated tree sequence: each
    contiguous node of a clade contributes its genomic span and its tsdate-estimated MRCA age."""
    iv = defaultdict(list)
    for e in dated1.edges():
        iv[e.parent].append((e.left, e.right))
    look = defaultdict(list)
    for n, cl in node_clade1.items():
        if len(cl) == 1 or n not in iv:
            continue
        age = dated1.node(n).time
        for l, r in _merge_intervals(iv[n]):
            look[cl].append((l, r, age))
    for cl in look:
        look[cl].sort()
    return look


def _assemble(pars, positions, seqlen, S, geno_at, assembly, pk, age_lookup=None):
    """Build a tskit tree sequence from per-position decoded trees (`pars`), assigning node identity by
    `assembly` (contiguous|global|bridge|bridge_noroot), then placing mutations by parsimony.  When `age_lookup` is
    given (refine 2nd pass), a bridge across a gap ALSO requires the two MRCA ages to agree (≤ AGE_TOL
    in log).  The tree changes at the next SNP, or at the midpoint between two SNPs with BREAKPOINTS 'midpoint'.
    Returns (ts, diag, node_clade)."""
    if BREAKPOINTS not in ("next_snp", "midpoint"):
        raise ValueError(f"unknown breakpoints {BREAKPOINTS}")
    if BREAKPOINTS == "midpoint" and any(float(x) % 1 for x in positions):
        raise ValueError("midpoint breakpoints need whole-number SNP positions")
    next_node = S
    node_clade = {i: frozenset([i]) for i in range(S)}       # node_id → clade; samples 0..S-1
    prev = {}; gmap = {}; last_seen = {}
    FULL = frozenset(range(S))                                # the (trivial) root clade = all samples
    # 'bridge_noroot' behaves exactly like 'bridge', EXCEPT the ROOT is keyed on its top split (its set
    # of child clades) instead of the always-present full-sample clade.  The full set is a clade in every
    # local tree, so clade-only identity collapses the root to ONE genome-wide node (→ one dated age →
    # a hard time ceiling + a huge pile-up).  Keying on the children makes recombination hand the root a
    # fresh identity whenever the top split changes, restoring the true continuum of root ages.
    bridgelike = "bridge" if assembly == "bridge_noroot" else assembly

    def assign(cl, cur, k, root_key=None):
        nonlocal next_node
        if len(cl) == 1:
            return next(iter(cl))
        key = root_key if (assembly == "bridge_noroot" and cl == FULL) else cl
        nd = cur.get(key)
        if nd is not None:
            return nd
        if bridgelike == "global":
            nd = gmap.get(key)
        elif bridgelike == "bridge":
            rec = last_seen.get(key)
            ok = rec is not None and (k - rec[0] - 1) * pk.get(len(cl), 0.0) < EPS
            if ok and age_lookup is not None:               # refine: ages on both sides must agree
                a_now = _age_at(age_lookup, cl, positions[k])
                a_prev = _age_at(age_lookup, cl, positions[rec[0]])
                ok = (a_now is not None and a_prev is not None and
                      abs(np.log(a_now + 1.0) - np.log(a_prev + 1.0)) < AGE_TOL)
            nd = rec[1] if ok else None
        else:                                               # contiguous
            nd = prev.get(key)
        if nd is None:
            nd = next_node; next_node += 1; node_clade[nd] = cl
            if bridgelike == "global":
                gmap[key] = nd
        cur[key] = nd
        if bridgelike == "bridge":
            last_seen[key] = (k, nd)
        return nd

    edge_rows = []
    run_ref, run_ref_cl, run_id = None, None, 0              # bridge_noroot: reference split (+clades) + run id
    for k, par in enumerate(pars):
        left = float(positions[k])
        right = float(positions[k + 1]) if k + 1 < len(positions) else float(seqlen)
        # With BREAKPOINTS 'midpoint', the tree changes at x + ceil((y - x) / 2) between the SNPs x and y, so the
        # first tree still starts at its SNP and the last still ends at seqlen.
        if BREAKPOINTS == "midpoint":
            if k > 0:
                left = float(positions[k - 1] + np.ceil((positions[k] - positions[k - 1]) / 2))
            if k + 1 < len(positions):
                right = float(positions[k] + np.ceil((positions[k + 1] - positions[k]) / 2))
        if right <= left:
            continue
        edges, clade = _clades_and_edges(par, S)
        root_key = None
        if assembly == "bridge_noroot":                     # root identity = its top split (child clades),
            cur_split = frozenset(cc for pc, cc in edges if pc == FULL)   # within ≤ ROOT_TOL switches of the ref
            cur_cl = frozenset(clade.values()) if ROOT_METRIC == "clade" else None
            if run_ref is None:
                d = 1 << 30                                 # first tree always starts a run
            elif ROOT_METRIC == "clade":                    # distance in CLADE switches (≈ recombinations)
                d = _clade_switch_dist(cur_split, cur_cl, run_ref, run_ref_cl)
            else:                                           # distance in LEAF switches
                d = _bipart_dist(cur_split, run_ref)
            if d > ROOT_TOL:
                run_id += 1                                 # split left the tolerance → new root node
                run_ref, run_ref_cl = cur_split, cur_cl     # reset reference to the run's first split
            elif ROOT_CMP == "adjacent":
                run_ref, run_ref_cl = cur_split, cur_cl     # same run: advance reference to track drift
            root_key = ("root", run_id)
        cur = {}
        for pc, cc in edges:
            edge_rows.append((left, right, assign(pc, cur, k, root_key), assign(cc, cur, k, root_key)))
        prev = cur

    # topology-only tree sequence (placeholder time = clade size → valid parent>child ordering)
    tables = tskit.TableCollection(sequence_length=seqlen)
    for n in range(next_node):
        cl = node_clade[n]
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE if len(cl) == 1 else 0,
                             time=0.0 if len(cl) == 1 else float(len(cl)))
    for left, right, p, c in edge_rows:
        tables.edges.add_row(left=left, right=right, parent=p, child=c)
    tables.sort()
    topo = tables.tree_sequence()

    # parsimony-map each SNP's mutation(s) onto its LOCAL marginal tree
    tables = topo.dump_tables()
    tables.sites.clear(); tables.mutations.clear()
    alleles = ["0", "1"]
    spos = sorted(geno_at)
    n_sites, extra, si = 0, 0, 0
    for tree in topo.trees():
        l, r = tree.interval.left, tree.interval.right
        while si < len(spos) and spos[si] < l:
            si += 1
        j = si
        while j < len(spos) and spos[j] < r:
            posn, g = spos[j], geno_at[spos[j]]
            j += 1
            if g.min() == g.max():
                continue
            anc, muts = tree.map_mutations(g, alleles)
            sid = tables.sites.add_row(position=posn, ancestral_state=anc)
            for m in muts:
                tables.mutations.add_row(site=sid, node=m.node, derived_state=m.derived_state)
            n_sites += 1
            extra += max(0, len(muts) - 1)
        si = j
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    ts = tables.tree_sequence()
    diag = dict(n_sites=n_sites, homoplasy_extra=extra,
                extra_per_site=(extra / n_sites if n_sites else float("nan")))
    return ts, diag, node_clade


def model_decode_to_ts(ckpt, true_trees_path, device="cpu", decoder="avg_constr", assembly=None, model=None):
    """Assemble the model's per-position decode into a tskit tree sequence with mutations placed by
    MAXIMUM PARSIMONY (tskit Tree.map_mutations / Hartigan) on each local tree — correct for ANY
    decoder, not just avg_constr.  Returns (ts, diag) where diag reports homoplasy: under avg_constr
    every site needs 1 mutation (carriers ARE a clade); otherwise parsimony adds extra mutations and
    `diag['extra_per_site']` measures how incompatible the topology is with the data.

    ARG assembly (GNN_BL_ASSEMBLY / `assembly`): node-sharing scheme for dating — an A/B knob.
      'contiguous': a clade keeps its node only across ADJACENT intervals; reappearing after a
            gap mints a NEW node (its own age), like a real ARG. Unbiased but pools fewer mutations/node.
      'global'   : a clade reuses ONE node wherever it appears (across gaps too). More mutation pooling
            (lower-variance times) but conflates distinct-age ancestors of a recurring clade.
      'bridge'   (default): size-aware middle (GNN_BL_EPS). Carry a clade across a gap iff a coincidental
            re-creation is unlikely — (#intervening trees) · p_k < EPS, where p_k = empirical chance
            that a specific k-leaf set is a clade (steep in k). Cherries (large p_k) → ~contiguous;
            rare large clades (tiny p_k) → ~global; the root (p_k=1) → contiguous.
    GNN_BL_REFINE_AGE (opt-in, bridge only): a 2nd pass — assemble contiguous, date it, then re-bridge
      requiring the clade's two MRCA ages to agree (≤ GNN_BL_AGE_TOL in log) as well as being rare. This
      is joint node-identity+time inference (SINGER-like); ~3× cost; helps most in the cherry regime.
    Placeholder node times = clade size (valid parent>child ordering); date_ts re-estimates them."""
    import torch
    torch.set_grad_enabled(False)
    assembly = assembly or BL_ASSEMBLY
    V.DECODER = decoder
    if model is None:                                         # load once and reuse across files; callers
        from model import infer_and_load      # (the runner) pass a pre-loaded model in
        model = infer_and_load(ckpt).to(device)
    true_ts = tskit.load(true_trees_path)
    S = true_ts.num_samples
    items = _flat_stream(true_ts, S)
    preds, _ = _sliding_predict(model, items, S, torch.device(device))
    ts, diag, _ = assemble_decoded(preds, items, S, true_ts.sequence_length, assembly=assembly)
    return ts, diag


# ── parallel per-position decode ────────────────────────────────────────────────
# The constrained-UPGMA decode of each site is independent, so we fan it out over a POOL of workers.
# We use a `fork` context and stash the (CPU-numpy) predictions/alleles in module globals set just
# before the pool is created: fork's copy-on-write lets every worker read the big prediction matrices
# WITHOUT pickling them per task — only the index goes out and the small parent array comes back.
# preds are detached to CPU numpy first so no worker ever touches CUDA (an inherited CUDA context is
# fine to hold but not to USE across a fork); the pure-numpy decode never does.
_DEC_PREDS = None            # list of S×S prediction matrices (CPU numpy)
_DEC_ALLELES = None          # list of length-S focal-genotype vectors (CPU numpy)
_DEC_S = None


def _decode_worker_setup(decoder):
    """Runs once per worker after fork: pin the decoder (decode_model reads train.DECODER) and keep each
    worker single-threaded so N processes don't oversubscribe the cores with BLAS threads."""
    V.DECODER = decoder
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _decode_worker(k):
    return k, np.asarray(decode_model(_DEC_PREDS[k], _DEC_S, alleles=_DEC_ALLELES[k]), dtype=np.int64)


def _to_np(a):
    import torch
    return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)


def _decode_all(preds, items, S, n_workers):
    """Decode every position → list of parent arrays (in position order).  n_workers<=1 → serial legacy
    path (bit-identical); else a fork pool.  0/negative → os.cpu_count()."""
    n = len(items)
    if n_workers is None or n_workers == 1 or n <= 1:
        return [decode_model(preds[k], S, alleles=items[k][0][:S, HAP_K]) for k in range(n)]
    if n_workers <= 0:
        n_workers = os.cpu_count() or 1
    n_workers = min(n_workers, n)

    global _DEC_PREDS, _DEC_ALLELES, _DEC_S
    _DEC_PREDS = [_to_np(preds[k]) for k in range(n)]
    _DEC_ALLELES = [_to_np(items[k][0])[:S, HAP_K] for k in range(n)]
    _DEC_S = S
    try:
        ctx = mp.get_context("fork")
        chunk = max(1, n // (n_workers * 4))            # amortise task dispatch, keep load balanced
        with ctx.Pool(n_workers, initializer=_decode_worker_setup, initargs=(V.DECODER,)) as pool:
            pars = [None] * n
            for k, par in pool.imap_unordered(_decode_worker, range(n), chunksize=chunk):
                pars[k] = par
        return pars
    finally:
        _DEC_PREDS = _DEC_ALLELES = _DEC_S = None       # release the COW references


def _squash_edges(ts, node_clade=None):
    """Lossless pre-dating edge compression (GNN_BL_SQUASH): merge a (parent,child) that persists over
    ADJACENT intervals into one wide-span edge via tskit simplify(keep_unary=True) — same topology, node
    set, sites and mutations, just far fewer edge rows (→ ~6× cheaper tsdate; edge count drives its cost).
    simplify RENUMBERS node IDs; every current consumer reads pairwise SAMPLE TMRCA (renumber-invariant),
    but if a node_clade map is passed we remap it through simplify's node_map so it stays correct too."""
    if node_clade is None:
        return ts.simplify(keep_unary=True, filter_sites=False, filter_populations=False,
                           filter_individuals=False, reduce_to_site_topology=False)
    ts2, nmap = ts.simplify(keep_unary=True, filter_sites=False, filter_populations=False,
                            filter_individuals=False, reduce_to_site_topology=False, map_nodes=True)
    nc = {int(nmap[o]): cl for o, cl in node_clade.items() if nmap[o] != tskit.NULL}
    return ts2, nc


def assemble_decoded(preds, items, S, seqlen, assembly=None):
    """Assemble per-position model predictions into a DATABLE tskit tree sequence — the shared core of
    the topology→ts step, independent of where `items` came from (simulated .trees or a VCF front-end).

    `preds`  : per-position predicted matrices (from _sliding_predict).
    `items`  : matching per-position (x, M, pos) tuples — x[:S, HAP_K] is the focal genotype used both
               for the avg_constr decode and for placing that site's mutation.  M is ignored here.
    Decodes each position with the CURRENT V.DECODER (caller sets it, e.g. "avg_constr"), assigns node
    identity by `assembly` (contiguous|global|bridge, using module EPS/REFINE_AGE), and parsimony-maps
    each site's mutation.  Returns (ts, diag, node_clade); node times are placeholders for date_ts."""
    assembly = assembly or BL_ASSEMBLY
    positions = [it[2] for it in items]

    # ── decode every position ONCE; collect genotypes; (bridge) tally clade sizes for p_k ──
    _t_decode = time.perf_counter()                          # sub-stage timing (for the runtime breakdown)
    pars = _decode_all(preds, items, S, DECODE_WORKERS)      # serial or fork-pool (GNN_DECODE_WORKERS)
    geno_at = {}                                             # position → genotype vector (0/1 per leaf)
    size_count = defaultdict(int)                           # clade size → #occurrences (for empirical p_k)
    for k, (x, tM, pos) in enumerate(items):
        geno_at[float(pos)] = (np.asarray(x[:S, HAP_K]) > 0).astype(np.int8)
        if assembly in ("bridge", "bridge_noroot"):
            _, clade = _clades_and_edges(pars[k], S)
            for cl in clade.values():
                if len(cl) >= 2:
                    size_count[len(cl)] += 1
    # p_k = P(a specific k-subset is a clade) ≈ (#size-k clades) / (#trees · C(S,k)); steep in k.
    ntrees = max(1, len(pars))
    pk = {k: size_count[k] / (ntrees * comb(S, k)) for k in size_count}
    _decode_s = time.perf_counter() - _t_decode              # per-position decode (+clade tally for bridge)

    # OPT-IN refine (bridge only): pass 1 = contiguous → date → per-clade ages; pass 2 = bridge that also
    # requires the two ages to agree before merging across a gap (joint node-identity + time).
    age_lookup = None
    if assembly == "bridge" and REFINE_AGE:
        ts1, _, nc1 = _assemble(pars, positions, seqlen, S, geno_at, "contiguous", pk)
        age_lookup = _build_age_lookup(nc1, date_ts(ts1, method="tsdate"))

    _t_asm = time.perf_counter()
    ts, diag, node_clade = _assemble(pars, positions, seqlen, S, geno_at, assembly, pk, age_lookup)
    diag["decode_s"] = _decode_s                             # node identity + edge build + parsimony mutation map
    diag["assemble_s"] = time.perf_counter() - _t_asm
    diag["n_edges"] = ts.num_edges
    if SQUASH:                                               # opt-in lossless edge compression → cheaper tsdate
        _t_sq = time.perf_counter()
        ts, node_clade = _squash_edges(ts, node_clade)
        diag["squash_s"] = time.perf_counter() - _t_sq
        diag["n_edges_squashed"] = ts.num_edges
    return ts, diag, node_clade


# ── runner: date each available topology and score vs truth ─────────────────────
def _new_pair_state(cap):
    """Reservoir state for a plain UNIFORM random sample of up to `cap` leaf pairs over the whole stream
    (every pair equally likely ⇒ the CSV is the natural pairwise-TMRCA distribution)."""
    return {"K": cap, "items": [], "seen": 0}


def _collect_pairs(true_ts, inf_by_tag, scenario, size, fi, state, stride):
    """Stream every leaf pair at strided positions into a single uniform reservoir.  Each kept row =
    (position, i, j, true_tmrca, each method's tmrca for that pair) — pairwise, so no NAs / no matching."""
    S = true_ts.num_samples
    I, J = np.triu_indices(S, 1)
    K, items = state["K"], state["items"]
    positions = [s.position for s in true_ts.sites()]
    for pos in positions[::max(1, stride)]:
        trueM = tmrca_matrix(true_ts.at(pos), S)
        mM = {}
        for tag, its in inf_by_tag.items():
            try:
                mM[tag] = tmrca_matrix(its.at(pos), S)
            except Exception:
                mM[tag] = None
        for p in range(len(I)):
            state["seen"] += 1
            if len(items) < K:
                slot = len(items)                           # append
            else:
                r = int(np.random.randint(state["seen"]))   # reservoir replace w.p. K/seen
                if r >= K:
                    continue
                slot = r
            i, j = int(I[p]), int(J[p])
            item = {"scenario": scenario, "size": size, "file": fi, "position": pos, "i": i, "j": j,
                    "true_tmrca": float(trueM[i, j])}
            for tag in _NODE_TAGS:
                m = mM.get(tag)
                item[tag] = float(m[i, j]) if m is not None else None
            if slot == len(items):
                items.append(item)
            else:
                items[slot] = item


def _write_node_csv(rows, path):
    import csv
    cols = ["scenario", "size", "file", "position", "i", "j", "true_tmrca"] + list(_NODE_TAGS)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            w.writerow(["" if r.get(c) is None else r.get(c) for c in cols])
    print(f"\n[pair CSV] wrote {len(rows)} leaf-pair rows → {path}", flush=True)


def _pair_binned_table(rows, nbins=8):
    """Per (scenario,size): pairwise-TMRCA accuracy in TRUE-TMRCA quantile bins — per method pearson/rmse/
    bias of log-tmrca.  Pairwise ⇒ every method has a value, so no recovery column needed."""
    cells = defaultdict(list)
    for r in rows:
        cells[(r["scenario"], r["size"])].append(r)
    print("\n══ pairwise-TMRCA accuracy by TRUE-TMRCA quantile (log-tmrca vs truth) ══", flush=True)
    for scen, size in sorted(cells):
        rs = cells[(scen, size)]
        tv = np.array([r["true_tmrca"] for r in rs], float)
        q = np.quantile(tv, np.linspace(0, 1, nbins + 1)); q[-1] = np.inf
        print(f"\n  {scen} S{size} ({len(rs)} pairs)")
        print(f"    {'true-tmrca bin':>22} {'method':>8} | {'pearson':>7} {'rmse_log':>8} {'bias_log':>8} {'n':>6}")
        for b in range(nbins):
            lo, hi = q[b], q[b + 1]
            binr = [r for r in rs if lo <= r["true_tmrca"] < hi]
            if not binr:
                continue
            label = f"[{lo:.0f},{'inf' if np.isinf(hi) else format(hi, '.0f')})"
            for tag in _NODE_TAGS:
                pres = [r for r in binr if r.get(tag) is not None and r[tag] > 0]
                if len(pres) < 3:
                    continue
                t = np.log(np.array([r["true_tmrca"] for r in pres]) + 1.0)
                m = np.log(np.array([r[tag] for r in pres]) + 1.0)
                pear = float(np.corrcoef(m, t)[0, 1]) if m.std() > 1e-9 and t.std() > 1e-9 else float("nan")
                print(f"    {label:>22} {tag:>8} | {pear:>7.4f} {float(np.sqrt(np.mean((m-t)**2))):>8.4f} "
                      f"{float(np.mean(m-t)):>8.4f} {len(pres):>6}")


def _date_one(scenario, size, file_indices, acc, model=None, device="cpu", pair_state=None, gacc=None):
    """Score each method's branch lengths vs truth, extending acc[method][metric] with raw values (and,
    if gacc is given, streaming every pair into gacc[method] for the global-pooled block).
    Each job is (tag, ts, mode, method):  mode='native' uses the method's OWN node times (Relate,
    SINGER — their own daters); mode='date' applies a dater to a topology (tsinfer→tsdate = its real
    pipeline; our model→GNN_DATING_METHOD; 'true+tsdate' = dater ceiling on the perfect topology)."""
    extra_per_site = []
    for fi in file_indices:
        base = os.path.join(DATA_DIR, f"{scenario}_S{size}_{fi}")
        true_path = base + ".trees"
        if not os.path.exists(true_path):
            continue
        true_ts = tskit.load(true_path)
        jobs = [("true", true_ts, "date", DATING_METHOD)]           # chosen dater on PERFECT topology (ceiling)
        for tag in ("relate", "singer"):                            # NATIVE branch lengths (own dater)
            fname = RELATE_FILE if tag == "relate" else tag         # relate → variable-Ne trees by default
            p = f"{base}_{fname}.trees"
            if not os.path.exists(p) and tag == "relate":           # fall back to const-Ne if _relate_vc absent
                p = f"{base}_relate.trees"
            if os.path.exists(p):
                jobs.append((tag, tskit.load(p), "native", None))
        p = f"{base}_tsinfer.trees"                                 # tsinfer has no real dates → tsdate it
        if os.path.exists(p):
            jobs.append(("tsinfer", tskit.load(p), "date", "tsdate"))
        if CKPT:                                                    # our topology + the chosen dater
            try:
                mts_model, diag = model_decode_to_ts(CKPT, true_path, model=model, device=device)
                jobs.append(("model", mts_model, "date", DATING_METHOD))
                extra_per_site.append(diag["extra_per_site"])
            except Exception as e:
                print(f"    [model assemble failed {scenario} S{size} f{fi}: {e}]", flush=True)
        inf_by_tag = {}
        for tag, ts, mode, method in jobs:
            try:
                inf = ts if mode == "native" else date_ts(ts, method=method)
                inf_by_tag[tag] = inf
                gp = gacc.setdefault(tag, _new_gpool()) if gacc is not None else None
                a = dating_accuracy(inf, true_ts, gpool=gp)
                for k, v in a.items():
                    acc.setdefault(tag, {}).setdefault(k, []).extend(v)
            except Exception as e:
                print(f"    [{tag} {scenario} S{size} f{fi}: {e}]", flush=True)
        if pair_state is not None:                                 # OPT-IN per-pair TMRCA collection
            _collect_pairs(true_ts, inf_by_tag, scenario, size, fi, pair_state, NODE_STRIDE)
    return float(np.mean(extra_per_site)) if extra_per_site else float("nan")


def _print_block(title, per_size, extra):
    print(f"\n=== {title} ===")
    print(f"  {'Size':>4} {'method':>8} | {'pearson_log':>11} {'spearman':>8} {'rmse_log':>8} {'bias_log':>8} {'n':>6}")
    for size in sorted(per_size):
        for tag in ("true", "relate", "tsinfer", "singer", "model"):
            if tag not in per_size[size]:
                continue
            s = summarize(per_size[size][tag])
            print(f"  {size:>4} {tag:>8} | {s['pearson_log']:>11.4f} {s['spearman']:>8.4f} "
                  f"{s['rmse_log']:>8.4f} {s['bias_log']:>8.4f} {s['n']:>6}")
        if size in extra and extra[size] == extra[size]:
            print(f"       {'(model homoplasy/site '+format(extra[size],'.3f')+')':>40}")


def _print_pair_block(title, per_size_g):
    """GLOBAL-POOLED pairwise TMRCA error (all pairs pooled, not per-position averaged) — exposes the
    catastrophic-tail pairs the per-position block dilutes.  med/p99 = |log-TMRCA err| (natural log)."""
    print(f"\n=== {title} — pairwise TMRCA error, GLOBAL-POOLED over all pairs ===")
    print(f"  {'Size':>4} {'method':>8} | {'pearson_log':>11} {'spearman':>8} {'rmse_log':>8} "
          f"{'bias_log':>8} {'n':>9}")
    for size in sorted(per_size_g):
        for tag in ("true", "relate", "tsinfer", "singer", "model"):
            if tag not in per_size_g[size]:
                continue
            s = _summarize_g(per_size_g[size][tag])
            print(f"  {size:>4} {tag:>8} | {s['pearson_log']:>11.4f} {s['spearman']:>8.4f} "
                  f"{s['rmse_log']:>8.4f} {s['bias_log']:>8.4f} {s['n']:>9}")


def main():
    n_test = max(1, V.FILES_PER // 6)
    files = lambda: list(range(max(0, V.FILES_PER - BL_FILES), V.FILES_PER))   # held-out tail
    print(f"branch-length eval | scenarios={BL_SCENARIOS} sizes={BL_SIZES} files={files()} "
          f"method={DATING_METHOD} relate_file={RELATE_FILE} mut_rate={MUT_RATE} ckpt={'yes' if CKPT else 'no'}"
          f"{' assembly='+BL_ASSEMBLY if CKPT else ''}{f' eps={EPS}' if CKPT and BL_ASSEMBLY in ('bridge','bridge_noroot') else ''}"
          f"{f' root_tol={ROOT_TOL} root_metric={ROOT_METRIC} root_cmp={ROOT_CMP}' if CKPT and BL_ASSEMBLY=='bridge_noroot' else ''}"
          f"{f' refine_age(tol={AGE_TOL})' if CKPT and BL_ASSEMBLY=='bridge' and REFINE_AGE else ''}"
          f"{' squash' if CKPT and SQUASH else ''} "
          f"(tsdate {'present' if tsdate else 'MISSING — pip install tsdate'})", flush=True)
    model, dev = None, "cpu"
    if CKPT:                                                       # load the model ONCE, reuse every file
        from model import infer_and_load
        import torch; torch.set_grad_enabled(False)
        dev = _resolve_device()
        model = infer_and_load(CKPT).to(dev)
        print(f"  model loaded once on {dev}", flush=True)
    pooled = {}                                                    # size -> method -> metric -> list
    pooled_g = {}                                                  # size -> method -> global-pool sums
    node_rows = [] if NODE_CSV else None                           # OPT-IN per-pair collection
    ncells = max(1, len(BL_SCENARIOS) * len(BL_SIZES))             # per-cell share of NODE_MAX (spread evenly)
    for scenario in BL_SCENARIOS:
        per_size, per_size_g, extra = {}, {}, {}
        for size in BL_SIZES:
            acc, gacc = {}, {}
            state = _new_pair_state(max(1, NODE_MAX // ncells)) if node_rows is not None else None
            extra[size] = _date_one(scenario, size, files(), acc, model=model, device=dev,
                                    pair_state=state, gacc=gacc)
            if state is not None:
                node_rows.extend(state["items"])
            if acc:
                per_size[size] = acc
                for tag, md in acc.items():                        # accumulate into ALL-pooled
                    for k, v in md.items():
                        pooled.setdefault(size, {}).setdefault(tag, {}).setdefault(k, []).extend(v)
            if gacc:
                per_size_g[size] = gacc
                for tag, g in gacc.items():                        # accumulate global-pool into ALL-pooled
                    _gpool_merge(pooled_g.setdefault(size, {}).setdefault(tag, _new_gpool()), g)
        if per_size:
            _print_block(f"scenario: {scenario}", per_size, extra)
            _print_pair_block(f"scenario: {scenario}", per_size_g)
    if len(BL_SCENARIOS) > 1 and pooled:
        _print_block("ALL scenarios pooled", pooled, {})
        _print_pair_block("ALL scenarios pooled", pooled_g)
    if node_rows is not None:                                      # OPT-IN: per-pair CSV + true-tmrca-binned table
        _write_node_csv(node_rows, NODE_CSV)
        _pair_binned_table(node_rows)
    print("\nBranch-length accuracy vs truth (higher pearson/spearman, lower rmse_log = better). NOTE each "
          "method uses its OWN dating: relate & singer = native node times; tsinfer = tsinfer+tsdate; "
          "model = our topology + GNN_DATING_METHOD (tsdate|polegon); 'true' = that dater on the PERFECT "
          "topology (its ceiling). Key reads: model vs tsinfer (isolates TOPOLOGY effect), model vs singer "
          "(bar to chase), and tsdate-vs-polegon = run twice and compare the 'true' and 'model' rows.", flush=True)


if __name__ == "__main__":
    main()
