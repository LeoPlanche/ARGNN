"""
simulate.py: Papuan archaic-introgression simulations with the true introgressed segments.

Demography: stdpopsim HomSap PapuansOutOfAfrica_10J19 (Jacobs et al. 2019), with two Denisovan pulses
(Den1 at about 1028 generations, Den2 at about 1576) into Papuans and one Neanderthal pulse (Nea1 into
Ghost, 1853 generations). The panel is that of the real-data ARG (12 Papuan, 5 YRI, 3 Neanderthal, and
1 Denisovan individuals, diploid), with the archaics as ancient samples at the model's sampling times
(NeaA 2612 generations, DenA 2058).

Truth: a census at 1900 generations (after the last pulse, before the DenA sampling time) records the
population of every lineage. A Papuan haplotype whose census ancestor is in Den1 or Den2 is
Denisovan-introgressed there, and in Nea1 Neanderthal-introgressed. Census nodes are then simplified away.

As in the real VCF build, sites where an archaic individual is heterozygous are dropped
(--keep-archaic-het disables this). The saved true ARG keeps the same sites as the VCF.

Outputs per replicate i in --out-dir:
  rep{i}.vcf.gz       phased VCF, individuals PAP0..PAP11, YRI0..YRI4, NEA0..NEA2, DEN0
  rep{i}_true.trees   true ARG, with sample names as node metadata {"name": "PAP3_hap1", ...}, as written
                      by src/vcf_to_trees.py
  rep{i}_truth.tsv    sample (PAP{j}_hap{k}), source (Denisovan or Neanderthal), start, end

Seeds start at 1,000,000.

Run:
  python -u archaic_introgression/simulations/simulate.py --out-dir OUT_DIR --reps 10 --length 20e6
"""
import argparse
import copy
import gzip
import os

import msprime
import numpy as np
import stdpopsim
import tskit

CENSUS_TIME = 1900
RECOMB_RATE = 1e-8
SOURCE_OF = {"Den1": "Denisovan", "Den2": "Denisovan", "Nea1": "Neanderthal"}
PREFIX = {"Papuan": "PAP", "YRI": "YRI", "NeaA": "NEA", "DenA": "DEN"}
SEED0 = 1_000_000


def demography():
    d = copy.deepcopy(stdpopsim.get_species("HomSap").get_demographic_model("PapuansOutOfAfrica_10J19").model)
    d.add_census(time=CENSUS_TIME)
    d.sort_events()
    return d


def true_segments(ts, d, papuan_nodes):
    """{papuan node: [(left, right, source), …]} from the census ancestors in Den1/Den2/Nea1."""
    pop_names = [p.name for p in d.populations]
    census = np.flatnonzero(ts.tables.nodes.flags & msprime.NODE_IS_CEN_EVENT)
    census_src = {int(c): SOURCE_OF.get(pop_names[ts.node(c).population]) for c in census}
    intro = [c for c, s in census_src.items() if s is not None]
    out = {int(n): [] for n in papuan_nodes}
    if intro:
        e = ts.tables.link_ancestors(samples=papuan_nodes, ancestors=intro)
        for l, r, p, c in zip(e.left, e.right, e.parent, e.child):
            out[int(c)].append((float(l), float(r), census_src[int(p)]))
    for c in out:                                         # merge abutting intervals of the same source
        segs = sorted(out[c]); m = []
        for l, r, s in segs:
            if m and m[-1][2] == s and l <= m[-1][1]:
                m[-1][1] = max(m[-1][1], r)
            else:
                m.append([l, r, s])
        out[c] = [tuple(x) for x in m]
    return out


def name_samples(ts):
    """Individual names (PAP0, YRI0, …) and per-sample-node names (<ind>_hap<k>)."""
    counts, ind_names, node_name = {}, [], {}
    for ind in ts.individuals():
        pop = ts.population(ts.node(ind.nodes[0]).population).metadata["name"]
        j = counts.get(pop, 0); counts[pop] = j + 1
        nm = f"{PREFIX[pop]}{j}"
        ind_names.append(nm)
        for k, n in enumerate(ind.nodes):
            node_name[int(n)] = f"{nm}_hap{k + 1}"
    return ind_names, node_name


def simulate_rep(i, args, d):
    seed = SEED0 + i
    ts = msprime.sim_ancestry(
        samples=[msprime.SampleSet(args.n_papuan, population="Papuan"),
                 msprime.SampleSet(args.n_yri, population="YRI"),
                 msprime.SampleSet(args.n_nea, population="NeaA", time=d["NeaA"].default_sampling_time),
                 msprime.SampleSet(args.n_den, population="DenA", time=d["DenA"].default_sampling_time)],
        demography=d, ploidy=2, recombination_rate=RECOMB_RATE, sequence_length=args.length, random_seed=seed)

    ind_names, node_name = name_samples(ts)
    papuan = [n for n, nm in node_name.items() if nm.startswith("PAP")]
    truth = true_segments(ts, d, papuan)

    ts = ts.simplify(keep_unary=False)                    # census nodes gone; samples keep IDs 0..S-1
    mts = msprime.sim_mutations(ts, rate=args.mu, model=msprime.BinaryMutationModel(), random_seed=seed)
    multi = [s.id for s in mts.sites() if len(s.mutations) > 1]
    n_het = 0
    if not args.keep_archaic_het:
        arch = [ind for ind, nm in zip(mts.individuals(), ind_names) if nm.startswith(("NEA", "DEN"))]
        G = mts.genotype_matrix()
        het = np.zeros(mts.num_sites, bool)
        for ind in arch:
            a, b = ind.nodes
            het |= G[:, a] != G[:, b]
        n_het = int(het.sum())
        multi = sorted(set(multi) | set(np.flatnonzero(het).tolist()))
    mts = mts.delete_sites(multi)

    tables = mts.dump_tables()                            # names as node metadata (like vcf_to_trees.py)
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    md = [({"name": node_name[n]} if n in node_name else {}) for n in range(tables.nodes.num_rows)]
    tables.nodes.packset_metadata([tables.nodes.metadata_schema.validate_and_encode_row(m) for m in md])
    mts = tables.tree_sequence()

    base = os.path.join(args.out_dir, f"rep{i}")
    mts.dump(base + "_true.trees")
    with gzip.open(base + ".vcf.gz", "wt") as fh:
        mts.write_vcf(fh, individual_names=ind_names, contig_id="1",
                      position_transform=lambda x: np.fmax(1, np.round(x)).astype(int))
    with open(base + "_truth.tsv", "w") as fh:
        fh.write("sample\tsource\tstart\tend\n")
        for n in sorted(truth):
            for l, r, s in truth[n]:
                fh.write(f"{node_name[n]}\t{s}\t{int(l)}\t{int(r)}\n")

    L = mts.sequence_length
    fr = {s: sum(r - l for segs in truth.values() for l, r, ss in segs if ss == s) / (L * len(papuan))
          for s in ("Denisovan", "Neanderthal")}
    print(f"rep{i} (seed {seed}): {mts.num_samples} haps, {mts.num_trees} trees, {mts.num_sites} SNPs "
          f"({n_het} archaic-het dropped) | true archaic per Papuan hap: Den {100*fr['Denisovan']:.2f}% "
          f"Nea {100*fr['Neanderthal']:.2f}%", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Papuan archaic-introgression simulations with true segments.")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "data"))
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--first-rep", type=int, default=0, help="index of the first replicate (seed = 1e6 + index)")
    ap.add_argument("--length", type=float, default=20e6, help="bp per replicate (default %(default)g)")
    ap.add_argument("--mu", type=float, default=1.25e-8)
    ap.add_argument("--n-papuan", type=int, default=12)
    ap.add_argument("--n-yri", type=int, default=5)
    ap.add_argument("--n-nea", type=int, default=3)
    ap.add_argument("--n-den", type=int, default=1)
    ap.add_argument("--keep-archaic-het", action="store_true", help="do NOT drop archaic-heterozygous sites")
    ap.add_argument("--force", action="store_true", help="re-simulate replicates that already exist")
    args = ap.parse_args()
    args.length = int(args.length)
    os.makedirs(args.out_dir, exist_ok=True)
    d = demography()
    for i in range(args.first_rep, args.first_rep + args.reps):
        base = os.path.join(args.out_dir, f"rep{i}")
        if not args.force and all(os.path.exists(base + s) for s in ("_true.trees", ".vcf.gz", "_truth.tsv")):
            print(f"rep{i}: exists, skipping", flush=True)
            continue
        simulate_rep(i, args, d)


if __name__ == "__main__":
    main()
