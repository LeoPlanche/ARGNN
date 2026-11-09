"""
vcf_to_trees.py — END-USER pipeline: a VCF in, a dated tskit .trees out.
========================================================================
This is the script an end user runs.  Given a phased VCF (.vcf or .vcf.gz) and a trained GNN
checkpoint, it:

  1. reads the VCF, keeps biallelic SNP sites, and SPLITS each phased sample into haplotype leaves
     (S = n_samples × ploidy); missing genotypes are filled as the ancestral/REF allele (0);
  2. runs the GNN over the haplotypes in sliding-window (full-context) mode and decodes a tree at
     every SNP with the DEFAULT `avg_constr` decoder (the focal SNP's split is forced to be an edge);
  3. assembles the per-SNP trees into one tskit tree sequence with the DEFAULT `bridge_noroot`
     node-sharing scheme (root kept from collapsing to a single genome-wide node, which would clamp
     all deep node ages to one ceiling value — tune with --root-tol/--root-metric/--root-cmp), with
     each change of tree at the midpoint between two SNPs (DEFAULT --breakpoints midpoint), and
     places each SNP's mutation on its carrier branch by parsimony;
  4. estimates node TIMES on that fixed topology with tsdate (`--mutation-rate` is REQUIRED — it sets
     the absolute time scale and must match your data);
  5. writes the dated .trees to OUTPUT.

The model arch is auto-inferred from the checkpoint; you only need to match the *feature* knobs that
change the model's INPUT — GNN_HAP_WINDOW (checked against the checkpoint) and GNN_SEQ_LEN (the
sliding-window length; set it to the value used at training for best results), plus GNN_KNN_K /
GNN_KNN_WEIGHTED / GNN_KNN_EPS if the training run used kNN input edges.

Usage:
  GNN_SEQ_LEN=256 GNN_HAP_WINDOW=... \
    conda run --no-capture-output -n lplancheenv python -u vcf_to_trees.py \
      INPUT.vcf.gz OUTPUT.trees --mutation-rate 1.25e-8 --ckpt gnn_v7_varsize_best.pt
  # subset to some samples (comma-list or a one-name-per-line file):
  #   … vcf_to_trees.py IN.vcf.gz OUT.trees --mutation-rate 1.25e-8 --samples HG001,HG002,HG003

Leaf identity: each haplotype leaf carries its source as NODE METADATA {"name": "<sample>_hap<k>"}
  (k 1-based over that sample's kept haplotypes) — read it back with ts.node(i).metadata["name"].
  Leaf node IDs are 0..S-1 in the order samples appear (or the --samples order), sample-major then
  haplotype-minor, after dropping all-missing haplotype columns (haploid padding).

Notes / assumptions:
  • Polarisation: REF is treated as ancestral (0), ALT as derived (1) — as in the training sims.  If
    your VCF is not ancestral-polarised, the decode still works (topology is largely flip-invariant)
    but interpret with care.
  • A .trees file is a single contig; if the VCF has several, pass --contig to choose one.
  • --samples restricts to the named samples (all their haplotypes, in the given order); sites that
    become monomorphic within the subset are dropped.
  • `--no-date` skips step 4 and writes topology only (placeholder node times = clade size).
"""
import argparse
import os
import sys

import numpy as np
import torch
import tskit

import train as V
from train import _sliding_predict, HAP_K, HAP_W
from gnn_topology import hap_window
from model import infer_and_load
import branch_length as BL


# ── VCF → haplotype genotype matrix ──────────────────────────────────────────────
def load_vcf_haplotypes(path, contig=None, keep_samples=None):
    """Read `path` (.vcf/.vcf.gz), keep biallelic SNPs on one contig, split phased genotypes into
    haplotype leaves, fill missing with ancestral (0).  If `keep_samples` is given (list of VCF sample
    names), only those samples are kept, IN THE ORDER GIVEN.  Returns a dict with:
      G         [S, n_var]  encoded ±1 (ancestral −1, derived +1), matching encode_genotype_matrix;
      positions [n_var]     strictly increasing site positions (VCF POS);
      names     [S]         per-leaf '<sample>_hap<k>' (k 1-based over that sample's kept haplotypes);
      contig, n_samples, n_samples_total, ploidy, n_raw, n_kept, n_dupdropped, n_missing_filled,
      n_hap_dropped, n_mono_dropped."""
    try:
        import allel
    except ImportError:
        raise SystemExit("scikit-allel is required to read VCFs: `pip install scikit-allel`")

    fields = ["samples", "variants/CHROM", "variants/POS", "variants/is_snp", "variants/numalt", "calldata/GT"]
    callset = allel.read_vcf(path, fields=fields, alt_number=1)
    if callset is None or "calldata/GT" not in callset:
        raise SystemExit(f"{path}: no genotype (GT) data found")

    sample_names = [str(s) for s in np.asarray(callset["samples"])]   # VCF column order
    chrom = np.asarray(callset["variants/CHROM"])
    pos   = np.asarray(callset["variants/POS"])
    gt    = callset["calldata/GT"]                                # [n_var, n_samples, ploidy]
    n_raw = len(pos)
    is_snp = callset.get("variants/is_snp", np.ones(n_raw, bool))
    numalt = callset.get("variants/numalt", np.ones(n_raw, int))

    contigs = sorted(set(chrom.tolist()))
    if contig is None:
        if len(contigs) > 1:
            raise SystemExit(f"VCF spans multiple contigs {contigs}; pass --contig to pick one "
                             f"(a .trees file is a single contig)")
        contig = contigs[0]
    elif contig not in contigs:
        raise SystemExit(f"--contig {contig!r} not in VCF (has {contigs})")

    keep = (chrom == contig) & np.asarray(is_snp, bool) & (np.asarray(numalt) == 1)   # biallelic SNPs
    if not keep.any():
        raise SystemExit(f"no biallelic SNP sites on contig {contig!r}")

    g = allel.GenotypeArray(gt[keep])                            # [n_kept, n_samples_total, ploidy]

    # Optional sample subset — keep ALL haplotypes of each chosen sample, in the REQUESTED order.
    if keep_samples:
        idx_of = {name: i for i, name in enumerate(sample_names)}
        missing = [s for s in keep_samples if s not in idx_of]
        if missing:
            raise SystemExit(f"--samples: {len(missing)} name(s) not found in VCF: "
                             f"{missing[:10]}{' …' if len(missing) > 10 else ''}")
        sel = [idx_of[s] for s in keep_samples]
        g = allel.GenotypeArray(np.asarray(g)[:, sel, :])
        chosen = list(keep_samples)
    else:
        chosen = sample_names

    n_samples, ploidy = g.shape[1], g.shape[2]
    # per-haplotype (sample, hap) labels in to_haplotypes() order == sample-major, hap-minor
    hap_labels = [(chosen[si], k) for si in range(n_samples) for k in range(ploidy)]
    H = np.asarray(g.to_haplotypes())                            # [n_kept, n_samples*ploidy], 0/1, missing = −1
    # scikit-allel reads GT at a FIXED ploidy (2 by default): a haploid VCF is padded with an
    # all-missing 2nd haplotype per sample.  Drop haplotype columns that are missing at EVERY site —
    # this removes those padding artifacts (and any genuinely all-missing sample, which carries no info).
    keep_hap = ~(H < 0).all(axis=0)
    n_hap_dropped = int((~keep_hap).sum())
    H = H[:, keep_hap]
    hap_labels = [lbl for lbl, k in zip(hap_labels, keep_hap) if k]
    n_missing = int((H < 0).sum())                              # remaining (real) missing calls
    H[H < 0] = 0                                                 # missing → ancestral

    p = pos[keep].astype(float)
    order = np.argsort(p, kind="stable")                        # VCF is usually sorted, but be safe
    p, H = p[order], H[order]
    uniq = np.concatenate([[True], np.diff(p) > 0])             # tskit needs unique site positions
    n_dup = int((~uniq).sum())
    p, H = p[uniq], H[uniq]

    # After any subset, a site polymorphic in the full VCF can be monomorphic here — drop such sites
    # (they carry no topology signal and would give avg_constr a degenerate focal split).
    col = H.sum(axis=1)
    seg = (col > 0) & (col < H.shape[1])
    n_mono = int((~seg).sum())
    p, H = p[seg], H[seg]

    G = H.T.astype(float)                                        # [S, n_var]
    G[G == 0] = -1.0                                             # ancestral −1, derived +1 (== encode_genotype_matrix)
    names = [f"{s}_hap{k + 1}" for (s, k) in hap_labels]         # k is the ORIGINAL hap index (0-based)
    return dict(G=G, positions=p, names=names, contig=contig, n_samples=n_samples,
                n_samples_total=len(sample_names), ploidy=ploidy, n_hap_dropped=n_hap_dropped,
                n_raw=n_raw, n_kept=int(keep.sum()), n_dupdropped=n_dup,
                n_missing_filled=n_missing, n_mono_dropped=n_mono)


def build_items(G, positions):
    """Per-SNP model inputs (x [N,W] node features, dummy M, position) — mirrors V._flat_stream but
    from a genotype matrix (no true tree needed at inference)."""
    S = G.shape[0]
    N = 2 * S - 1
    M_dummy = torch.zeros((S, S))                               # target matrix, unused at inference
    items = []
    for vi in range(G.shape[1]):
        win = hap_window(G, vi)                                 # [S, W], W = 2*HAP_K+1
        x = np.concatenate([win, np.zeros((N - S, win.shape[1]))], axis=0)
        items.append((torch.tensor(x, dtype=torch.float), M_dummy, float(positions[vi])))
    return items


def _parse_samples(arg):
    """--samples value → ordered, de-duplicated list of sample names.  If `arg` is an existing file,
    read one name per line (blank lines and #comments ignored); otherwise split on commas.  None→None."""
    if not arg:
        return None
    if os.path.isfile(arg):
        with open(arg) as fh:
            raw = [ln.strip() for ln in fh]
        raw = [x for x in raw if x and not x.startswith("#")]
    else:
        raw = [x.strip() for x in arg.split(",") if x.strip()]
    seen, out = set(), []
    for x in raw:                                              # keep first occurrence, drop duplicates
        if x not in seen:
            seen.add(x); out.append(x)
    if not out:
        raise SystemExit(f"--samples {arg!r}: no sample names parsed")
    return out


def _annotate_names(ts, names):
    """Attach per-leaf '<sample>_hap<k>' identity as node metadata {'name': …} on the sample nodes
    (leaves 0..S-1, in order); internal nodes get empty metadata.  Returns a new tree sequence."""
    tables = ts.dump_tables()
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    meta, si = [], 0
    for nid in range(tables.nodes.num_rows):
        if tables.nodes.flags[nid] & tskit.NODE_IS_SAMPLE:
            meta.append({"name": names[si]}); si += 1
        else:
            meta.append({})
    if si != len(names):
        raise SystemExit(f"internal: {si} sample nodes but {len(names)} names — cannot label")
    tables.nodes.packset_metadata(
        [tables.nodes.metadata_schema.validate_and_encode_row(m) for m in meta])
    return tables.tree_sequence()


def _check_hap_window(ckpt_path):
    """The checkpoint's input feature width fixes HAP_W; error clearly if GNN_HAP_WINDOW mismatches."""
    sd = torch.load(ckpt_path, map_location="cpu")
    type_dim    = sd["type_embed.weight"].shape[1] if "type_embed.weight" in sd else 0
    node_id_dim = sd["node_id_embed.weight"].shape[1] if "node_id_embed.weight" in sd else 0
    feat = sd["node_input_proj.weight"].shape[1] - type_dim - node_id_dim
    if feat != HAP_W:
        raise SystemExit(f"checkpoint hap window W={feat} != current HAP_W={HAP_W}; "
                         f"re-run with GNN_HAP_WINDOW={(feat - 1) // 2}")


def main():
    ap = argparse.ArgumentParser(description="VCF → GNN topology → tsdate → tskit .trees")
    ap.add_argument("input", help="input VCF (.vcf or .vcf.gz)")
    ap.add_argument("output", help="output tskit .trees")
    ap.add_argument("--mutation-rate", type=float, required=True,
                    help="per-base per-generation mutation rate for tsdate (REQUIRED; sets the time scale)")
    ap.add_argument("--ckpt", default=os.environ.get("GNN_CKPT", V.SAVE_PATH), help="model checkpoint")
    ap.add_argument("--contig", default=None, help="contig/chromosome to use if the VCF has several")
    ap.add_argument("--samples", default=None,
                    help="restrict to these VCF samples (all their haplotypes, in this order): either a "
                         "comma-list 'HG001,HG002' or a path to a file with one sample name per line")
    ap.add_argument("--ne", type=int, default=BL.NE, help="effective pop size prior for tsdate (default %(default)s)")
    ap.add_argument("--decoder", default="avg_constr", help="decode mode (default avg_constr)")
    ap.add_argument("--assembly", default=os.environ.get("GNN_BL_ASSEMBLY", "bridge_noroot"),
                    choices=["bridge_noroot", "bridge", "contiguous", "global"],
                    help="node-sharing scheme across trees (default bridge_noroot: keeps the root from "
                         "collapsing to one genome-wide node, which would clamp all deep ages to a ceiling)")
    ap.add_argument("--eps", type=float, default=BL.EPS, help="bridge tolerance (default %(default)s; higher → more bridging)")
    # bridge_noroot root-sharing knobs (recommended defaults 1 / clade / anchor); env-overridable.
    ap.add_argument("--root-tol", type=int, default=int(os.environ.get("GNN_BL_ROOT_TOL", "1")),
                    help="keep the same root node while the top split stays within this tolerance (default %(default)s)")
    ap.add_argument("--root-metric", default=os.environ.get("GNN_BL_ROOT_METRIC", "clade"),
                    choices=["clade", "leaf"], help="--root-tol unit (default %(default)s)")
    ap.add_argument("--root-cmp", default=os.environ.get("GNN_BL_ROOT_CMP", "anchor"),
                    choices=["anchor", "adjacent"], help="--root-tol reference split (default %(default)s)")
    ap.add_argument("--breakpoints", default=os.environ.get("GNN_BL_BREAKPOINTS", "midpoint"),
                    choices=["midpoint", "next_snp"], help="where the tree changes between the SNPs x and y: "
                    "midpoint = x + ceil((y - x) / 2), next_snp = y (default %(default)s)")
    ap.add_argument("--dating-method", default="tsdate", help="tsdate | polegon (default tsdate)")
    ap.add_argument("--no-date", action="store_true", help="skip dating; write topology only (times = clade size)")
    ap.add_argument("--save-embeddings", default=None, help="ALSO save the per-SNP LEAF EMBEDDINGS "
                    "(the model's final per-leaf representation, before the pairwise head builds the tree) "
                    "to this .npz path. Non-default; for probing (PCA / local-ancestry).")
    ap.add_argument("--embed-stride", type=int, default=1, help="with --save-embeddings, keep every Nth SNP "
                    "position (default 1 = all; raise to shrink the file — embeddings are [n_pos, S, H])")
    ap.add_argument("--embed-space", default="P", choices=["L", "P", "both"], help="which leaf embedding(s) "
                    "to save: L=contextual pre-metric [.,.,H] (richer/raw), P=metric space [.,.,D] whose "
                    "pairwise distances ARE the tree (Euclidean-calibrated, better for PCA/NN), both (default %(default)s)")
    ap.add_argument("--device", default=os.environ.get("GNN_DEVICE", ""), help="'', 'cpu', 'cuda'…")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"input not found: {args.input}")
    torch.set_grad_enabled(False)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    BL.EPS = args.eps                                           # honoured inside branch_length._assemble
    BL.ROOT_TOL, BL.ROOT_METRIC, BL.ROOT_CMP = args.root_tol, args.root_metric, args.root_cmp  # bridge_noroot
    BL.BREAKPOINTS = args.breakpoints                           # read inside branch_length._assemble

    # 1. VCF → haplotypes
    keep_samples = _parse_samples(args.samples)
    print(f"[1/4] reading {args.input}", flush=True)
    vcf = load_vcf_haplotypes(args.input, contig=args.contig, keep_samples=keep_samples)
    G, positions, names = vcf["G"], vcf["positions"], vcf["names"]
    S = G.shape[0]
    if S < 3:
        raise SystemExit(f"need ≥3 haplotype leaves, got S={S}")
    if G.shape[1] < 2:
        raise SystemExit(f"need ≥2 usable SNP sites, got {G.shape[1]}")
    seqlen = float(positions[-1] + 1.0)
    drop_note = f", {vcf['n_hap_dropped']} all-missing hap-cols dropped" if vcf["n_hap_dropped"] else ""
    subset_note = (f"{vcf['n_samples']}/{vcf['n_samples_total']} samples (subset)" if keep_samples
                   else f"{vcf['n_samples']} samples")
    mono_note = f", {vcf['n_mono_dropped']} now-monomorphic sites dropped" if vcf["n_mono_dropped"] else ""
    print(f"      contig={vcf['contig']} | {subset_note} × ploidy {vcf['ploidy']}{drop_note} "
          f"→ {S} haplotype leaves | {vcf['n_kept']}/{vcf['n_raw']} biallelic-SNP sites kept "
          f"({vcf['n_dupdropped']} dup-pos dropped, {vcf['n_missing_filled']} missing→ancestral{mono_note}) | "
          f"span {seqlen/1e6:.3f} Mb", flush=True)

    # 2. model
    _check_hap_window(args.ckpt)
    V.DECODER = args.decoder
    print(f"[2/4] loading model {args.ckpt} (decoder={V.DECODER}, seq_len={V.SEQ_LEN}, "
          f"knn_k={V.KNN_K}) on {device}", flush=True)
    model = infer_and_load(args.ckpt).to(device)
    if S > model.num_samples:
        raise SystemExit(f"VCF has S={S} haplotypes but the checkpoint supports at most "
                         f"{model.num_samples} (trained max); subsample or retrain for larger S")

    items = build_items(G, positions)
    asm_note = (f"{args.assembly} root_tol={args.root_tol}/{args.root_metric}/{args.root_cmp}"
                if args.assembly == "bridge_noroot" else args.assembly)
    print(f"[3/4] running model over {len(items)} SNPs + assembling ({asm_note}, avg_constr, "
          f"breakpoints {args.breakpoints})", flush=True)
    if args.save_embeddings:
        preds, _, embeds, win_ids = _sliding_predict(model, items, S, device, embed_space=args.embed_space)
        k = list(range(0, len(win_ids), max(1, args.embed_stride)))         # subsampled positions
        save = dict(positions=np.asarray([positions[i] for i in k], dtype=np.int64),
                    leaves=np.asarray(names),
                    window_id=np.asarray([win_ids[i] for i in k], dtype=np.int32),  # source forward-window/pos
                    seq_len=np.int64(V.SEQ_LEN), slide_margin=np.int64(V.SLIDE_MARGIN))
        shapes = []
        for sp, lst in embeds.items():                                      # sp in {L, P}
            arr = np.stack([lst[i].numpy() for i in k]).astype(np.float16)  # [n_pos, S, H_sp]
            save[f"embeddings_{sp}"] = arr
            shapes.append(f"{sp}={arr.shape}")
        np.savez_compressed(args.save_embeddings, **save)
        print(f"      saved leaf embeddings [{', '.join(shapes)}] (n_pos,S,H) + window_id "
              f"({len(np.unique(save['window_id']))} forward-windows, seq_len={V.SEQ_LEN}, "
              f"margin={V.SLIDE_MARGIN}, stride {args.embed_stride}, float16) → {args.save_embeddings}", flush=True)
    else:
        preds, _ = _sliding_predict(model, items, S, device)
    ts_topo, diag, _ = BL.assemble_decoded(preds, items, S, seqlen, assembly=args.assembly)
    print(f"      {ts_topo.num_trees} trees, {ts_topo.num_nodes} nodes, {ts_topo.num_sites} sites | "
          f"homoplasy {diag['extra_per_site']:.3f} extra-mut/site "
          f"(≈0 expected under avg_constr)", flush=True)

    # 4. dating
    if args.no_date:
        out_ts = ts_topo
        print("[4/4] --no-date: writing topology only (placeholder node times)", flush=True)
    else:
        print(f"[4/4] dating with {args.dating_method} (mutation_rate={args.mutation_rate}, Ne={args.ne})", flush=True)
        try:
            out_ts = BL.date_ts(ts_topo, method=args.dating_method,
                                mutation_rate=args.mutation_rate, Ne=args.ne)
        except Exception as e:
            import traceback
            traceback.print_exc()                              # surface the REAL cause (e.g. bare AssertionError)
            raise SystemExit(f"dating failed ({type(e).__name__}: {e}). Re-run with --no-date for topology "
                             f"only, or check --mutation-rate / that tsdate is installed. Note: the clade-based "
                             f"archaic scan needs only topology, so --no-date is fine for it.")

    out_ts = _annotate_names(out_ts, names)                    # leaf node metadata {'name': '<sample>_hapN'}
    out_ts.dump(args.output)
    print(f"wrote {args.output}: {out_ts.num_trees} trees, {out_ts.num_nodes} nodes, "
          f"{out_ts.num_sites} sites, leaf names as node metadata"
          f"{'' if args.no_date else ', dated node times'}", flush=True)


if __name__ == "__main__":
    main()
