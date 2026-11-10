"""
compare_segments.py: base-pair overlap between two archaic-segment call sets.

    precision = L_overlap / L_inferred      recall = L_overlap / L_truth
    enrichment = precision / (L_truth / L_accessible)
    Jaccard = L_overlap / (L_inferred + L_truth - L_overlap)      F1 = harmonic mean of precision and recall

Enrichment is how much more of the calls land on the comparison set than random placement in the
accessible genome would. "truth" can itself be another caller (e.g. DAIseg); Jaccard and F1 do not depend
on which side is called truth.

Both sides are read as (individual, start, end[, label]) and compared per individual. Haplotype suffixes
(_hap1, _hap2) are stripped and the two haplotypes are unioned, because the DAIseg map is diploid.
Everything is intersected with --mask first.

The calls, the comparison set, and the mask must be on the same genome build; nothing here checks that.

Run:
  python -u archaic_introgression/compare_segments.py CALLS.tsv TRUTH.txt \
      --chrom 1 --mask callable_chr1.bed --mask-chrom 1
  # restrict either side by its label column: --truth-label Denisovan --calls-label Denisovan
"""
import argparse
import os
import re

HAP_RE = re.compile(r"_hap\d+$")

# ── interval algebra on sorted lists of half-open [s,e) ──────────────────────────
def merge(iv):
    """Sort and union a list of (start, end)."""
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def intersect(a, b):
    """Intersection of two MERGED interval lists."""
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def total(iv):
    return sum(e - s for s, e in iv)


# ── readers ─────────────────────────────────────────────────────────────────────
def load_bed(path, chrom=None):
    """BED of accessible/callable regions -> merged intervals.  `chrom` filters column 1 (accepting
    both '21' and 'chr21' spellings); None keeps every line, which is right for a single-chrom BED."""
    iv = []
    want = {chrom, chrom.replace("chr", ""), "chr" + chrom.replace("chr", "")} if chrom else None
    for line in open(path):
        p = line.split()
        if len(p) < 3 or not p[1].isdigit():
            continue
        if want is not None and p[0] not in want:
            continue
        iv.append((int(p[1]), int(p[2])))
    if not iv:
        raise SystemExit(f"{path}: no intervals (wrong --mask-chrom? got {chrom!r})")
    return merge(iv)


def load_segments(path, chrom=None, label=None):
    """A scan TSV or a DAIseg map -> {individual: merged intervals}, plus
    {individual: {label: merged intervals}}.  Columns are found by name."""
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        cols = {c.strip().lower(): i for i, c in enumerate(header)}

        def pick(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            return None

        ci, cs, ce = pick("sample", "individual", "name"), pick("start"), pick("end")
        clab, cchrom = pick("ancestry", "source"), pick("chrom", "chromosome")
        if ci is None or cs is None or ce is None:
            raise SystemExit(f"{path}: need sample/individual + start + end columns, got {header}")

        by_ind, by_lab = {}, {}
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) <= max(x for x in (ci, cs, ce) if x is not None):
                continue
            if cchrom is not None and chrom is not None:
                if p[cchrom].replace("chr", "") != chrom.replace("chr", ""):
                    continue
            lab = p[clab] if clab is not None else "NA"
            if label is not None and lab != label:
                continue
            ind = HAP_RE.sub("", p[ci])                      # HGDP00549_hap1 -> HGDP00549
            s, e = int(float(p[cs])), int(float(p[ce]))
            by_ind.setdefault(ind, []).append((s, e))
            by_lab.setdefault(ind, {}).setdefault(lab, []).append((s, e))

    if not by_ind:
        raise SystemExit(f"{path}: no segments survived the chrom/label filter")
    return ({k: merge(v) for k, v in by_ind.items()},
            {k: {l: merge(x) for l, x in d.items()} for k, d in by_lab.items()})


# ── main ────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="bp-overlap precision/recall/enrichment between two call sets.")
    ap.add_argument("calls", help="calls (a TSV from outgroup_archaic_scan.py)")
    ap.add_argument("truth", help="comparison set (a DAIseg map, or another TSV of calls)")
    ap.add_argument("--mask", required=True, help="BED of accessible/callable regions IN THE SAME BUILD")
    ap.add_argument("--mask-chrom", default=None, help="chromosome to keep from --mask (e.g. chr21 or 21)")
    ap.add_argument("--chrom", default=None, help="chromosome to keep from either input that has a chrom column")
    ap.add_argument("--calls-label", default=None, help="keep only calls with this source/ancestry")
    ap.add_argument("--truth-label", default=None, help="keep only truth with this source/ancestry")
    ap.add_argument("--per-individual", action="store_true", help="also print the per-individual table")
    ap.add_argument("--plot", default=None, help="positional-overlap SVG (default none)")
    ap.add_argument("--bin", type=float, default=100000, help="bin size for --plot (default %(default)s)")
    args = ap.parse_args()

    mask = load_bed(args.mask, args.mask_chrom)
    calls, _ = load_segments(args.calls, args.chrom, args.calls_label)
    truth, truth_lab = load_segments(args.truth, args.chrom, args.truth_label)

    print(f"mask:  {len(mask)} intervals, {total(mask)/1e6:.2f} Mb accessible")
    print(f"calls: {os.path.basename(args.calls)} — {len(calls)} individuals"
          + (f", label={args.calls_label}" if args.calls_label else ""))
    print(f"truth: {os.path.basename(args.truth)} — {len(truth)} individuals"
          + (f", label={args.truth_label}" if args.truth_label else ""))

    shared = sorted(set(calls) & set(truth))
    if not shared:
        raise SystemExit(f"no individuals in common.\n  calls: {sorted(calls)[:5]}\n  truth: {sorted(truth)[:5]}")
    only_c, only_t = sorted(set(calls) - set(truth)), sorted(set(truth) - set(calls))
    if only_c or only_t:
        print(f"  NOTE {len(shared)} individuals compared; calls-only {only_c}; truth-only {only_t}")

    # labels present in truth, so we can report which archaic our calls actually land on
    labels = sorted({l for d in truth_lab.values() for l in d} - {"NA"})

    L_ov = L_inf = L_tru = 0
    ov_by_lab = {l: 0 for l in labels}
    rows = []
    for ind in shared:
        a = intersect(calls[ind], mask)
        b = intersect(truth[ind], mask)
        o = intersect(a, b)
        L_ov += total(o); L_inf += total(a); L_tru += total(b)
        for l in labels:
            if l in truth_lab[ind]:
                ov_by_lab[l] += total(intersect(a, intersect(truth_lab[ind][l], mask)))
        rows.append((ind, total(a), total(b), total(o)))

    acc = total(mask) * len(shared)                      # accessible bp summed over individuals
    prec = L_ov / L_inf if L_inf else float("nan")
    rec = L_ov / L_tru if L_tru else float("nan")
    base = L_tru / acc if acc else float("nan")
    enr = prec / base if base else float("nan")
    uni = L_inf + L_tru - L_ov                           # bp in either call set (per individual, summed)
    jac = L_ov / uni if uni else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")

    print(f"\n{'':2}individuals compared     {len(shared)}")
    print(f"  L_inferred              {L_inf/1e6:9.3f} Mb   ({100*L_inf/acc:5.2f}% of accessible)")
    print(f"  L_truth                 {L_tru/1e6:9.3f} Mb   ({100*base:5.2f}% of accessible)")
    print(f"  L_overlap               {L_ov/1e6:9.3f} Mb")
    print(f"  precision               {100*prec:9.2f}%")
    print(f"  recall                  {100*rec:9.2f}%")
    print(f"  enrichment              {enr:9.2f}x   (precision / {100*base:.2f}% random baseline)")
    print(f"  F1                      {100*f1:9.2f}%")
    print(f"  Jaccard                 {100*jac:9.2f}%   (L_overlap / L_union, union {uni/1e6:.3f} Mb)")
    if labels:
        print("  overlap by truth label:")
        for l in labels:
            sh = ov_by_lab[l] / L_ov if L_ov else float("nan")
            print(f"    {l:<14} {ov_by_lab[l]/1e6:8.3f} Mb  ({100*sh:5.1f}% of overlap)")

    if args.per_individual:
        print(f"\n  {'individual':<12} {'called Mb':>10} {'truth Mb':>9} {'overlap Mb':>11} {'prec':>7} {'recall':>7}")
        for ind, ai, bi, oi in rows:
            print(f"  {ind:<12} {ai/1e6:10.3f} {bi/1e6:9.3f} {oi/1e6:11.3f} "
                  f"{100*oi/ai if ai else float('nan'):6.1f}% {100*oi/bi if bi else float('nan'):6.1f}%")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        end = mask[-1][1]
        nb = int(end // args.bin) + 1
        cc, tt = np.zeros(nb), np.zeros(nb)
        for ind in shared:
            for iv, acc_arr in ((intersect(calls[ind], mask), cc), (intersect(truth[ind], mask), tt)):
                for s, e in iv:
                    for k in range(int(s // args.bin), int(e // args.bin) + 1):
                        lo, hi = max(s, k * args.bin), min(e, (k + 1) * args.bin)
                        if hi > lo:
                            acc_arr[k] += hi - lo
        x = np.arange(nb) * args.bin / 1e6
        denom = args.bin * len(shared)
        fig, ax = plt.subplots(figsize=(12, 3.2))
        ax.fill_between(x, 100 * tt / denom, color="#999999", alpha=.55, label=f"truth ({os.path.basename(args.truth)})")
        ax.plot(x, 100 * cc / denom, color="#d62728", lw=.9, label=f"calls ({os.path.basename(args.calls)})")
        ax.set_xlabel("Position (Mb)"); ax.set_ylabel("% of individuals called")
        ax.set_title(f"precision {100*prec:.1f}%  recall {100*rec:.1f}%  enrichment {enr:.1f}x")
        ax.legend(fontsize=7); ax.grid(True, alpha=.2)
        fig.tight_layout(); fig.savefig(args.plot, bbox_inches="tight")
        print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
