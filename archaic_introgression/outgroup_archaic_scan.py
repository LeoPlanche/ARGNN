"""
outgroup_archaic_scan.py: topology-only scan for archaic introgression, polarised by an outgroup.

Needs an ARG that contains archaic samples and an African outgroup (e.g. Yoruba). For each target
(e.g. Papuan) haplotype h in a local tree, climb from h to the first ancestor whose subtree holds an
archaic or an outgroup sample:

  * the node holds an archaic and no outgroup: h joins the archaic before Africa, so h is flagged as
    introgressed, and the archaic clade under the node gives the source;
  * the node holds an outgroup: h joins Africa first, so h is not flagged. This also covers ancestral
    sharing, because an allele also present in Africans puts an outgroup sample in the node.

The test reads no node times. A tract carried by every target is still flagged, because the carrier
clade joins the archaic before the non-carrying outgroup.

Flagged trees are merged across gaps < --max-gap, and a segment is kept if it is at least --min-seg bp
long and has at least --min-snps flagged trees. Its source is the majority over its flagged trees
(Neanderthal, Denisovan, or ambiguous). The carrier count is the number of targets under the joining
node. Sample names are read from the node metadata that src/vcf_to_trees.py writes.

Run:
  python -u archaic_introgression/outgroup_archaic_scan.py IN.trees \
      --neanderthal Altai,Vindija,Chagyrskaya --denisovan Denisova --outgroup NA18
"""
import argparse, os
import numpy as np
import tskit


def scan(ts, target, neand, deni, outgr):
    """Per (tree, target hap): flag (archaic-before-outgroup), source (1=Nea,2=Den,0=ambig),
    carrier = # targets under the joining node.  Returns flag[nt,S], src[nt,S], carrier[nt,S], L, R."""
    S, nt = len(target), ts.num_trees
    flag = np.zeros((nt, S), dtype=bool); src = np.zeros((nt, S), dtype=np.int8)
    carrier = np.zeros((nt, S), dtype=np.int32)
    L = np.empty(nt); R = np.empty(nt)
    itN = ts.trees(tracked_samples=neand); itD = ts.trees(tracked_samples=deni)
    itO = ts.trees(tracked_samples=outgr); itT = ts.trees(tracked_samples=target)
    for ti, (tN, tD, tO, tT) in enumerate(zip(itN, itD, itO, itT)):
        L[ti], R[ti] = tN.interval.left, tN.interval.right
        for si, h in enumerate(target):
            a = h
            while a != tskit.NULL:
                nN = tN.num_tracked_samples(a); nD = tD.num_tracked_samples(a)
                nO = tO.num_tracked_samples(a)
                if nN or nD or nO:                     # first ancestor holding an archaic OR outgroup
                    if (nN or nD) and not nO:          # archaic present, outgroup ABSENT => introgression
                        flag[ti, si] = True
                        src[ti, si] = 1 if (nN and not nD) else (2 if (nD and not nN) else 0)
                        carrier[ti, si] = tT.num_tracked_samples(a)   # # targets sharing it here
                    break
                a = tN.parent(a)
    return flag, src, carrier, L, R


def _callable_bp(a_bp, b_bp, mask, ends):
    """bp of [a_bp,b_bp) covered by the sorted, merged callable-mask interval list."""
    import bisect
    tot = 0
    for ms, me in mask[bisect.bisect_right(ends, a_bp):]:
        if ms >= b_bp:
            break
        tot += min(b_bp, me) - max(a_bp, ms)
    return tot


def segments(flag, srow, crow, L, R, max_gap, min_seg, min_snps, mask=None, mask_frac=0.5):
    """Merged segments for one target hap. [(start,end,n_trees,min_carrier,mean_carrier,source_int), …].
    If `mask` (sorted merged CALLABLE intervals) is given, a segment is KEPT WHOLE only if its callable
    fraction ≥ mask_frac (a per-position mappability mask is too fine-grained to clip to — clipping
    would shred every segment below min_seg; instead we drop segments sitting mostly in non-callable
    low-mappability regions, which is where the false positives live)."""
    ends = [m[1] for m in mask] if mask is not None else None
    nt = len(flag); runs = []; i = 0
    while i < nt:
        if flag[i]:
            j = i
            while j + 1 < nt and flag[j + 1]:
                j += 1
            runs.append([L[i], R[j], i, j]); i = j + 1
        else:
            i += 1
    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] < max_gap:
            merged[-1][1], merged[-1][3] = run[1], run[3]
        else:
            merged.append(run)
    out = []
    for a_bp, b_bp, ti0, ti1 in merged:
        if b_bp - a_bp < min_seg:
            continue
        if mask is not None and _callable_bp(a_bp, b_bp, mask, ends) < mask_frac * (b_bp - a_bp):
            continue                                          # mostly non-callable → drop
        m = flag[ti0:ti1 + 1]
        if int(m.sum()) < min_snps:
            continue
        c = crow[ti0:ti1 + 1][m]; s = srow[ti0:ti1 + 1][m]
        n1, n2 = int((s == 1).sum()), int((s == 2).sum())
        source = 1 if n1 > n2 else (2 if n2 > n1 else 0)
        out.append((a_bp, b_bp, int(m.sum()), int(c.min()), float(c.mean()), source))
    return out


def main():
    ap = argparse.ArgumentParser(description="Outgroup-polarised archaic-introgression scan (topology-only).")
    ap.add_argument("trees")
    ap.add_argument("--neanderthal", default="Altai,Vindija,Chagyrskaya", help="Neanderthal name substrings (comma-sep)")
    ap.add_argument("--denisovan", default="Denisova", help="Denisovan name substrings (comma-sep)")
    ap.add_argument("--outgroup", default="NA18", help="African-outgroup name substrings (comma-sep, e.g. Yoruba IDs)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-seg", type=float, default=20000)
    ap.add_argument("--min-snps", type=int, default=20)
    ap.add_argument("--max-gap", type=float, default=5000)
    ap.add_argument("--mask", default=None, help="BED of CALLABLE regions (in the TREES' build, e.g. hg19); "
                    "drop segments whose callable fraction < --mask-frac (removes calls in low-mappability regions)")
    ap.add_argument("--mask-frac", type=float, default=0.5, help="min callable fraction to keep a segment (default %(default)s)")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    mask = None
    if args.mask:
        iv = []
        for line in open(args.mask):
            p = line.split()
            if len(p) >= 3 and p[1].isdigit():
                iv.append((int(p[1]), int(p[2])))
        iv.sort(); mask = [list(iv[0])]
        for s, e in iv[1:]:
            if s <= mask[-1][1]: mask[-1][1] = max(mask[-1][1], e)
            else: mask.append([s, e])
        mask = [tuple(m) for m in mask]
        print(f"mask: {len(mask)} callable intervals, {sum(e-s for s,e in mask)/1e6:.1f} Mb, dropping segments below --mask-frac", flush=True)

    ts = tskit.load(args.trees)
    names = {i: (ts.node(i).metadata.get("name", str(i)) if isinstance(ts.node(i).metadata, dict) else str(i))
             for i in ts.samples()}
    def match(pats): return [i for i in ts.samples() if any(p in names[i] for p in pats.split(","))]
    neand, deni, outgr = match(args.neanderthal), match(args.denisovan), match(args.outgroup)
    excl = set(neand + deni + outgr)
    target = [i for i in ts.samples() if i not in excl]
    if not (neand or deni) or not outgr or not target:
        raise SystemExit(f"need archaics + outgroup + targets: neand={len(neand)} deni={len(deni)} "
                         f"outgroup={len(outgr)} target={len(target)}")
    print(f"loaded {args.trees}: {len(target)} target haps, {len(outgr)} outgroup, "
          f"{len(neand)} Neanderthal + {len(deni)} Denisovan archaic haps, {ts.num_trees} trees, "
          f"{ts.sequence_length/1e6:.2f} Mb", flush=True)

    flag, src, carrier, L, R = scan(ts, target, neand, deni, outgr)
    out_path = args.out or (os.path.splitext(args.trees)[0] + "_outarchaic.tsv")
    SRC = {0: "ambiguous", 1: "Neanderthal", 2: "Denisovan"}
    rows, per_hap_bp = [], np.zeros(len(target))
    for si in range(len(target)):
        for a_bp, b_bp, ntr, mn, me, source in segments(flag[:, si], src[:, si], carrier[:, si], L, R,
                                                        args.max_gap, args.min_seg, args.min_snps, mask, args.mask_frac):
            rows.append((names[target[si]], si, SRC[source], int(a_bp), int(b_bp), int(b_bp - a_bp), ntr, mn, me))
            per_hap_bp[si] += b_bp - a_bp
    with open(out_path, "w") as fh:
        fh.write("sample\thap_id\tsource\tstart\tend\tlength\tn_trees\tmin_carrier\tmean_carrier\n")
        for r in sorted(rows, key=lambda x: (x[1], x[3])):
            fh.write("%s\t%d\t%s\t%d\t%d\t%d\t%d\t%d\t%.1f\n" % r)

    def cnt(n): return sum(1 for r in rows if r[2] == n)
    def mb(n): return sum(r[5] for r in rows if r[2] == n) / 1e6
    frac = per_hap_bp / ts.sequence_length
    print(f"wrote {out_path}: {len(rows)} segments "
          f"({cnt('Neanderthal')} Neanderthal / {cnt('Denisovan')} Denisovan / {cnt('ambiguous')} ambiguous)", flush=True)
    print(f"  flagged Mb: Neanderthal {mb('Neanderthal'):.2f} / Denisovan {mb('Denisovan'):.2f} / ambiguous {mb('ambiguous'):.2f}"
          f"  (Den:Nea = {mb('Denisovan')/max(mb('Neanderthal'),1e-9):.1f}:1)", flush=True)
    print(f"  mean per-haplotype flagged fraction: {100*frac.mean():.2f}%", flush=True)

    plot_path = args.plot if args.plot is not None else os.path.splitext(out_path)[0] + ".svg"
    if plot_path:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        S = len(target); fig, ax = plt.subplots(figsize=(11, 0.32 * S + 1.2))
        col = {"Neanderthal": "#1f77b4", "Denisovan": "#d62728", "ambiguous": "#999999"}
        for si in range(S):
            for sn in ("Neanderthal", "Denisovan", "ambiguous"):
                segs = [(r[3], r[5]) for r in rows if r[1] == si and r[2] == sn]
                if segs: ax.broken_barh(segs, (si - 0.4, 0.8), facecolors=col[sn], edgecolors="none")
        ax.set_xlim(0, ts.sequence_length); ax.set_ylim(-1, S)
        ax.set_yticks(range(S)); ax.set_yticklabels([names[target[i]] for i in range(S)], fontsize=6)
        ax.set_xlabel("Position (bp)"); ax.set_title(f"Outgroup-polarised archaic segments — {os.path.basename(args.trees)}")
        ax.legend(handles=[Patch(color=col[k], label=k) for k in col], loc="upper right", fontsize=8)
        ax.grid(True, axis="x", alpha=0.2); fig.tight_layout(); fig.savefig(plot_path, bbox_inches="tight")
        print(f"wrote {plot_path}", flush=True)


if __name__ == "__main__":
    main()
