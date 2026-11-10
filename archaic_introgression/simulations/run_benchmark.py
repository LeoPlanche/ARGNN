"""
run_benchmark.py: archaic-introgression detection on simulated Papuans, scored against the true segments.

Each step is skipped when its output exists, so a killed job can be resubmitted:

  1. simulate   simulate.py: data/rep{i}.vcf.gz, rep{i}_true.trees, rep{i}_truth.tsv
  2. infer      src/vcf_to_trees.py on every VCF, once per --ckpt: data/rep{i}__<ckpt-stem>.trees
                (topology only by default, the detector reads no times), and tsinfer with --tsinfer
  3. detect     outgroup_archaic_scan.scan and segments, on the true ARG and every inferred ARG:
                results/calls/<arg>__mg<max-gap>.tsv
  4. score      bp precision, recall, and F1 against the true segments: results/summary.csv

Scores are per haplotype, with the diploid union (haplotypes merged per individual) alongside. The
summary also has per-source scores, source-label accuracy, recall by carrier frequency and by tract
length, and the F1 spread across replicates.

The detector is run per SNP: each site takes the flag of its local tree, and flagged runs span the
midpoints between SNPs. So --min-snps counts SNPs on the true ARG too, which has far fewer trees than SNPs.

Run from the repo root (the results in ../results/simulations used these flags):
  python -u archaic_introgression/simulations/run_benchmark.py --tsinfer \
      --ckpt archaic_introgression/checkpoints/gnn_v7_varsize_rk0.5L3_knn7w_best_truesplit.pt,archaic_introgression/checkpoints/gnn_pap_out_finetuned.pt
"""
import argparse
import csv
import os
import re
import subprocess
import sys

import numpy as np
import tskit

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHAIC = os.path.dirname(HERE)
ROOT = os.path.dirname(ARCHAIC)
sys.path.insert(0, ARCHAIC)
from outgroup_archaic_scan import scan, segments                 # noqa: E402  (the real-data detector)
from compare_segments import merge, intersect, total             # noqa: E402

SRC = {0: "ambiguous", 1: "Neanderthal", 2: "Denisovan"}
# Graph and decoding settings that a checkpoint does not store: those of the reported runs. The tree changes
# at the next SNP (next_snp), the only behaviour when these results were made; src/vcf_to_trees.py now defaults
# to midpoint.
MODEL_ENV = {"GNN_KNN_K": "7", "GNN_KNN_WEIGHTED": "1", "GNN_SEQ_LEN": "256", "GNN_DECODER": "avg_constr",
             "GNN_BL_ASSEMBLY": "bridge_noroot", "GNN_BL_ROOT_TOL": "1", "GNN_BL_ROOT_METRIC": "clade",
             "GNN_BL_ROOT_CMP": "anchor", "GNN_BL_BREAKPOINTS": "next_snp"}
FREQ_BINS = [(1, 1), (2, 3), (4, 8), (9, 16), (17, 10**9)]
LEN_BINS = [(0, 20e3), (20e3, 50e3), (50e3, 100e3), (100e3, 200e3), (200e3, 1e12)]


# ── 1-2. simulate + infer ─────────────────────────────────────────────────────────
def run(cmd, env=None, log=None):
    print("  $ " + " ".join(cmd), flush=True)
    with open(log, "w") if log else open(os.devnull, "w") as fh:
        r = subprocess.run(cmd, env=env, stdout=fh if log else None, stderr=subprocess.STDOUT if log else None)
    if r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode})" + (f", see {log}" if log else ""))


def infer(vcf, out, ckpt, args):
    env = dict(os.environ)
    for k, v in MODEL_ENV.items():
        env.setdefault(k, v)                              # an explicit env var still wins
    cmd = [sys.executable, "-u", os.path.join(ROOT, "src", "vcf_to_trees.py"), vcf, out,
           "--mutation-rate", str(args.mu), "--ckpt", ckpt]
    if not args.date:
        cmd.append("--no-date")
    if args.device:
        cmd += ["--device", args.device]
    run(cmd, env=env, log=out + ".log")


def run_tsinfer(mts):
    """tsinfer tree sequence of mts, simplified to the samples, with 0 as the ancestral allele and without
    monomorphic sites."""
    import tsinfer
    with tsinfer.SampleData(sequence_length=mts.sequence_length) as sd:
        for v in mts.variants():
            g = (np.asarray(v.genotypes) != 0).astype(np.int8)
            if g.min() == g.max():
                continue
            sd.add_site(v.site.position, g, ["0", "1"], ancestral_allele=0)
    return tsinfer.infer(sd).simplify()                   # samples 0..S-1


def infer_tsinfer(true_path, out):
    """tsinfer baseline on the same SNPs the VCF holds (the true ARG keeps exactly the VCF's site set;
    REF = ancestral, as vcf_to_trees assumes).  Topology only — no tsdate, like our --no-date ARGs.
    Sample nodes get the same name metadata so the detector's name matching works."""
    true = tskit.load(true_path)
    its = run_tsinfer(true)                               # simplified → samples 0..S-1 in the true order
    tables = its.dump_tables()
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    md = [({"name": true.node(n).metadata["name"]} if n < true.num_samples else {})
          for n in range(tables.nodes.num_rows)]
    tables.nodes.packset_metadata([tables.nodes.metadata_schema.validate_and_encode_row(m) for m in md])
    tables.tree_sequence().dump(out)
    print(f"  tsinfer: {its.num_trees} trees, {its.num_sites} sites → {out}", flush=True)


# ── 3. detect ───────────────────────────────────────────────────────────────────
def detect(ts, max_gap, args):
    """Real-data detector on `ts`, evaluated per SNP.  Returns [(hap_name, source, start, end)]."""
    names = {i: ts.node(i).metadata.get("name", str(i)) if isinstance(ts.node(i).metadata, dict) else str(i)
             for i in ts.samples()}
    def match(pat): return [i for i in ts.samples() if any(p in names[i] for p in pat.split(","))]
    neand, deni, outgr = match(args.neanderthal), match(args.denisovan), match(args.outgroup)
    excl = set(neand + deni + outgr)
    target = [i for i in ts.samples() if i not in excl]
    if not (neand or deni) or not outgr or not target:
        raise SystemExit(f"name matching failed: neand={len(neand)} deni={len(deni)} outgroup={len(outgr)}")
    flag, src, carrier, _, _ = scan(ts, target, neand, deni, outgr)

    pos = ts.tables.sites.position
    ti = np.searchsorted(ts.breakpoints(as_array=True), pos, side="right") - 1    # tree index of each SNP
    mid = (pos[1:] + pos[:-1]) / 2
    L = np.concatenate([[0.0], mid]); R = np.concatenate([mid, [ts.sequence_length]])
    flag, src, carrier = flag[ti], src[ti], carrier[ti]

    rows = []
    for si, h in enumerate(target):
        for a, b, _, _, _, s in segments(flag[:, si], src[:, si], carrier[:, si], L, R,
                                         max_gap, args.min_seg, args.min_snps):
            rows.append((names[h], SRC[s], int(a), int(b)))
    return rows


# ── 4. score ────────────────────────────────────────────────────────────────────
def load_truth(path):
    tr = {}
    with open(path) as fh:
        next(fh)
        for line in fh:
            h, s, a, b = line.rstrip("\n").split("\t")
            tr.setdefault(h, []).append((int(a), int(b), s))
    return tr


def by_label(rows_by_hap, label=None):
    return {h: merge([(a, b) for a, b, s in v if label is None or s == label]) for h, v in rows_by_hap.items()}


def prf(ov, inf, tru):
    p = ov / inf if inf else float("nan"); r = ov / tru if tru else float("nan")
    f = 2 * p * r / (p + r) if inf and tru and (p + r) > 0 else (0.0 if inf or tru else float("nan"))
    return p, r, f


class Tally:
    """bp counts summed over (replicate, haplotype) units."""
    def __init__(self):
        self.ov = self.inf = self.tru = 0

    def add(self, calls, truth):
        o = total(intersect(calls, truth))
        self.ov += o; self.inf += total(calls); self.tru += total(truth)

    def prf(self):
        return prf(self.ov, self.inf, self.tru)


def score_rep(calls_rows, truth, haps, seq_len, acc):
    """Accumulate one replicate into `acc` (dict of Tally / arrays).  `haps` = all target hap names."""
    C = {h: [] for h in haps}
    for h, s, a, b in calls_rows:
        C[h].append((a, b, s))
    T = {h: truth.get(h, []) for h in haps}
    c_all, t_all = by_label(C), by_label(T)
    rep = Tally()
    for h in haps:
        acc["hap"].add(c_all[h], t_all[h]); rep.add(c_all[h], t_all[h])
        for lab in ("Denisovan", "Neanderthal"):
            c_l, t_l = by_label({h: C[h]}, lab)[h], by_label({h: T[h]}, lab)[h]
            acc[lab].add(c_l, t_l)
        # source accuracy: bp where call and truth overlap, label agreement
        for lab_t in ("Denisovan", "Neanderthal"):
            tt = by_label({h: T[h]}, lab_t)[h]
            for lab_c in ("Denisovan", "Neanderthal", "ambiguous"):
                acc["conf"][(lab_t, lab_c)] += total(intersect(by_label({h: C[h]}, lab_c)[h], tt))
    acc["rep_f1"].append(rep.prf()[2])

    ind = sorted({re.sub(r"_hap\d+$", "", h) for h in haps})       # diploid union
    for i in ind:
        hs = [h for h in haps if h.startswith(i + "_hap")]
        cc, tt = merge([iv for h in hs for iv in c_all[h]]), merge([iv for h in hs for iv in t_all[h]])
        acc["dip"].add(cc, tt)

    # recall by carrier frequency (n target haps introgressed at the tract midpoint) and by tract length
    for h in haps:
        for a, b in t_all[h]:
            m = (a + b) / 2
            f = sum(1 for g in haps if any(x <= m < y for x, y in t_all[g]))
            o = total(intersect(c_all[h], [(a, b)]))
            for k, (lo, hi) in enumerate(FREQ_BINS):
                if lo <= f <= hi:
                    acc["freq_ov"][k] += o; acc["freq_tr"][k] += b - a
            for k, (lo, hi) in enumerate(LEN_BINS):
                if lo <= b - a < hi:
                    acc["len_ov"][k] += o; acc["len_tr"][k] += b - a
    acc["bp_total"] += seq_len * len(haps)


def new_acc():
    return {"hap": Tally(), "dip": Tally(), "Denisovan": Tally(), "Neanderthal": Tally(),
            "conf": {(t, c): 0 for t in ("Denisovan", "Neanderthal") for c in ("Denisovan", "Neanderthal", "ambiguous")},
            "rep_f1": [], "freq_ov": np.zeros(len(FREQ_BINS)), "freq_tr": np.zeros(len(FREQ_BINS)),
            "len_ov": np.zeros(len(LEN_BINS)), "len_tr": np.zeros(len(LEN_BINS)), "bp_total": 0}


def summarise(arg, mg, acc):
    p, r, f = acc["hap"].prf(); dp, dr, df = acc["dip"].prf()
    cf = acc["conf"]
    right = cf[("Denisovan", "Denisovan")] + cf[("Neanderthal", "Neanderthal")]
    labelled = sum(v for (t, c), v in cf.items() if c != "ambiguous")
    rowd = {"arg": arg, "max_gap": int(mg), "precision": p, "recall": r, "f1": f,
            "f1_rep_mean": float(np.nanmean(acc["rep_f1"])), "f1_rep_sd": float(np.nanstd(acc["rep_f1"])),
            "dip_precision": dp, "dip_recall": dr, "dip_f1": df,
            "called_frac": acc["hap"].inf / acc["bp_total"], "true_frac": acc["hap"].tru / acc["bp_total"]}
    for lab, k in (("Denisovan", "den"), ("Neanderthal", "nea")):
        rowd[f"{k}_precision"], rowd[f"{k}_recall"], rowd[f"{k}_f1"] = acc[lab].prf()
        tot = sum(v for (t, c), v in cf.items() if t == lab)
        rowd[f"{k}_label_acc"] = cf[(lab, lab)] / tot if tot else float("nan")
    rowd["source_acc"] = right / labelled if labelled else float("nan")
    rowd["ambiguous_frac"] = 1 - labelled / sum(cf.values()) if sum(cf.values()) else float("nan")
    for k, (lo, hi) in enumerate(FREQ_BINS):
        rowd[f"recall_freq_{lo}-{hi if hi < 1e9 else 'max'}"] = acc["freq_ov"][k] / acc["freq_tr"][k] if acc["freq_tr"][k] else float("nan")
    for k, (lo, hi) in enumerate(LEN_BINS):
        rowd[f"recall_len_{int(lo/1e3)}-{int(hi/1e3) if hi < 1e12 else 'max'}kb"] = acc["len_ov"][k] / acc["len_tr"][k] if acc["len_tr"][k] else float("nan")
    return rowd


def plot(rows, path, mg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]   # fixed order
    sel = [r for r in rows if r["max_gap"] == mg]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(sel)); w = 0.26
    for k, (m, lab) in enumerate((("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"))):
        a1.bar(x + (k - 1) * w, [r[m] for r in sel], w - 0.02, color=COL[k], label=lab)
    a1.errorbar(x + w, [r["f1"] for r in sel], yerr=[r["f1_rep_sd"] for r in sel], fmt="none", ecolor="#52514e", lw=1)
    a1.set_xticks(x); a1.set_xticklabels([r["arg"] for r in sel], fontsize=8, rotation=15)
    a1.set_ylim(0, 1.15); a1.set_ylabel("per-haplotype bp score")
    a1.legend(fontsize=8, frameon=False, ncol=3, loc="upper center")
    a1.set_title(f"Detection vs true segments (max-gap {mg/1e3:.0f} kb)", fontsize=10)
    fk = [k for k in sel[0] if k.startswith("recall_freq_")]
    for i, r in enumerate(sel):
        a2.plot(range(len(fk)), [r[k] for k in fk], "-o", lw=2, ms=6, color=COL[i % len(COL)], label=r["arg"])
    a2.set_xticks(range(len(fk))); a2.set_xticklabels([k.replace("recall_freq_", "") for k in fk], fontsize=8)
    a2.set_xlabel("true-tract carriers among Papuan haplotypes"); a2.set_ylabel("recall"); a2.set_ylim(0, 1)
    a2.set_title("Recall by introgression frequency", fontsize=10); a2.legend(fontsize=8, frameon=False)
    for a in (a1, a2):
        a.grid(True, axis="y", alpha=0.25); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Simulation benchmark of archaic-introgression detection (true segments).")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "output", "data"))
    ap.add_argument("--results-dir", default=os.path.join(HERE, "output", "results"))
    ap.add_argument("--ckpt", default="", help="comma-separated checkpoints to infer ARGs with (empty → true ARG only)")
    ap.add_argument("--reps", type=int, default=10, help="replicates to simulate/use (default %(default)s)")
    ap.add_argument("--length", type=float, default=20e6, help="bp per replicate (default %(default)g)")
    ap.add_argument("--mu", type=float, default=1.25e-8)
    ap.add_argument("--date", action="store_true", help="also date the inferred ARGs with tsdate (not needed)")
    ap.add_argument("--device", default="", help="device for vcf_to_trees (default: a GPU if available)")
    ap.add_argument("--no-true-arg", action="store_true", help="skip the detector on the true ARG")
    ap.add_argument("--tsinfer", action="store_true", help="also benchmark tsinfer-inferred ARGs (baseline)")
    ap.add_argument("--neanderthal", default="NEA"); ap.add_argument("--denisovan", default="DEN")
    ap.add_argument("--outgroup", default="YRI")
    ap.add_argument("--min-seg", type=float, default=20000)
    ap.add_argument("--min-snps", type=int, default=20)
    ap.add_argument("--max-gaps", default="5000,10000", help="max-gap values to sweep (default %(default)s)")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.results_dir, "calls"), exist_ok=True)
    reps = list(range(args.reps))
    # 1. simulate (simulate.py itself skips existing replicates)
    run([sys.executable, "-u", os.path.join(HERE, "simulate.py"), "--out-dir", args.data_dir,
         "--reps", str(args.reps), "--length", str(args.length), "--mu", str(args.mu)])

    # 2. infer
    args_list = [] if args.no_true_arg else [("true", "true")]
    for ck in [c for c in args.ckpt.split(",") if c]:
        if not os.path.exists(ck):
            raise SystemExit(f"checkpoint not found: {ck}")
        stem = os.path.splitext(os.path.basename(ck))[0]
        args_list.append((stem, stem))
        for i in reps:
            out = os.path.join(args.data_dir, f"rep{i}__{stem}.trees")
            if os.path.exists(out):
                print(f"rep{i} {stem}: inferred ARG exists, skipping", flush=True)
                continue
            infer(os.path.join(args.data_dir, f"rep{i}.vcf.gz"), out, ck, args)
    if args.tsinfer:
        args_list.append(("tsinfer", "tsinfer"))
        for i in reps:
            out = os.path.join(args.data_dir, f"rep{i}__tsinfer.trees")
            if os.path.exists(out):
                print(f"rep{i} tsinfer: inferred ARG exists, skipping", flush=True)
                continue
            infer_tsinfer(os.path.join(args.data_dir, f"rep{i}_true.trees"), out)

    # 3-4. detect + score
    rows = []
    mgs = [float(m) for m in args.max_gaps.split(",")]
    for label, suffix in args_list:
        accs = {mg: new_acc() for mg in mgs}
        calls_all = {mg: [] for mg in mgs}
        for i in reps:
            ts = tskit.load(os.path.join(args.data_dir, f"rep{i}_{'true' if suffix == 'true' else '_' + suffix}.trees"))
            truth = load_truth(os.path.join(args.data_dir, f"rep{i}_truth.tsv"))
            haps = [ts.node(n).metadata["name"] for n in ts.samples() if ts.node(n).metadata["name"].startswith("PAP")]
            for mg in mgs:
                c = detect(ts, mg, args)
                calls_all[mg] += [(i,) + r for r in c]
                score_rep(c, truth, haps, ts.sequence_length, accs[mg])
        for mg in mgs:
            with open(os.path.join(args.results_dir, "calls", f"{label}__mg{int(mg)}.tsv"), "w") as fh:
                fh.write("rep\tsample\tsource\tstart\tend\n")
                for r in calls_all[mg]:
                    fh.write("\t".join(map(str, r)) + "\n")
            rows.append(summarise(label, mg, accs[mg]))
            r = rows[-1]
            print(f"{label:<45} mg={int(mg):>6}  P {100*r['precision']:5.1f}%  R {100*r['recall']:5.1f}%  "
                  f"F1 {100*r['f1']:5.1f}% (reps {100*r['f1_rep_mean']:.1f}±{100*r['f1_rep_sd']:.1f})  "
                  f"| diploid F1 {100*r['dip_f1']:5.1f}%  | source acc {100*r['source_acc']:5.1f}%  "
                  f"| called {100*r['called_frac']:.2f}% / true {100*r['true_frac']:.2f}%", flush=True)

    out_csv = os.path.join(args.results_dir, "summary.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {out_csv}", flush=True)
    plot(rows, os.path.join(args.results_dir, "f1_summary.pdf"), mgs[0])


if __name__ == "__main__":
    main()
