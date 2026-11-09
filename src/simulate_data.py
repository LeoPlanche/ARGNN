"""
simulate_data.py: the true tree sequences of the training, validation, and testing data
=======================================================================================
Simulates on the diploid time scale: msprime.sim_ancestry with ploidy=2, so every population size counts
diploid individuals, and each sample is one haplotype (msprime.SampleSet(k, ploidy=1)).
generate_varsize_data.py simulates with ploidy=1, which acts as if every population size were halved.
The rates, the size of const, MIN_SITES, and the mutation step are those of generate_varsize_data.py, copied
here so that this script needs no other script of src/: mutate adds mutations at MUT_RATE, then removes every
site that more than one mutation hit (msprime's genome is discrete), so that each site that remains carries one
mutation and is biallelic. The removal is in the tree sequence, so a VCF written from it has only these sites.

Models:
  const        one population of 10,000
  split        A (20,000) and B (5,000) split from ANC (10,000) 10,000 generations ago, half the haplotypes
               from each
  bott         one population of 10,000, reduced to 500 between 500 and 700 generations ago
  ooa          stdpopsim HomSap OutOfAfricaArchaicAdmixture_5R19, a third of the haplotypes from each of YRI,
               CEU, and CHB, the remainder from CHB, and Neanderthal and ArchaicAFR are not sampled
  zigzag_1s14  stdpopsim HomSap Zigzag_1S14, every haplotype from its one population
  papuans_10j19  stdpopsim HomSap PapuansOutOfAfrica_10J19: GNN_PAP_NEA haplotypes from NeaA, GNN_PAP_DEN
                 from DenA, GNN_PAP_YRI from YRI, and the rest from Papuan, as simulate_pap of
                 generate_varsize_data.py samples them. The defaults are the panel of the real-data run, 2 NeaA
                 (one Neanderthal), 2 DenA (one Denisovan), and 24 YRI, so 24 Papuan at 52 haplotypes.

Every haplotype is sampled at the sampling time of its population in msprime: 0, except NeaA (2,612 generations)
and DenA (2,058 generations) in papuans_10j19.

Seeds: seed = base + 1,000,000 x (index of the model in MODELS) + 1,000 x size + replicate, with one base per
data set and purpose (SEED_BASES), 100,000,000 apart. So no two seeds can be equal, and all lie above the seeds
of generate_varsize_data.py. The testing data also get a tool seed for the external tools, which gives a tool its
own seed in place of a simulation seed (msprime, Relate, and SINGER all use the Mersenne Twister, so one seed would
give a simulation and a tool the same random numbers). run_external_tools.py reads it from the table and gives it
to Relate and SINGER. A new model goes at the end of MODELS, so that the seeds of the others stay the same.

Checks, which end the run with an error after the table is written: the haplotypes of the sampling scheme, each
at the sampling time of its population, the sequence length, one mutation at every site, at least MIN_SITES sites
kept, no seed used twice, and, for each model and size, the mean over replicates of the mean pairwise coalescence
time of the true trees within a factor of TIME_SCALE_TOLERANCE of msprime's expectation. Under Snakemake, a failed
job loses its outputs, the table included, so only the printed lines remain in its log.

Outputs in GNN_VS_DIR, for each model and size:
  {model}_S{size}_{replicate}.trees  replicates 0 to GNN_VS_FILES - 1
  {model}_S{size}_simulations.csv    one row per replicate:
    model, size, replicate
    ancestry_seed, mutation_seed     the seeds of msprime.sim_ancestry and msprime.sim_mutations
    tool_seed                        (testing data only) the seed of the external tools, unused here
    sequence_length                  in base pairs
    removed_var_sites                sites hit by more than one mutation, removed by mutate
    qced_var_sites                   sites kept, each with one mutation
    trees                            marginal trees
    pi_site                          TreeSequence.diversity(mode="site") of the sites kept: the mean number of
                                     differences between two haplotypes per base pair
    pi_branch                        MUT_RATE times TreeSequence.diversity(mode="branch"): the diversity that the
                                     true trees predict
    t2_branch                        TreeSequence.diversity(mode="branch") / 2: the mean pairwise coalescence time
                                     of the true trees, in generations
    t2_msprime                       msprime's expected mean pairwise coalescence time under the model and the
                                     sampling scheme (Demography.debug().mean_coalescence_time), in generations.
                                     Every lineage starts at time 0, so for papuans_10j19 it ignores the sampling
                                     times of NeaA and DenA and is about 0.2% too high at the default panel.
    relate_ne                        round(pi_branch / (2 MUT_RATE)): Relate's -N, the effective population size
                                     of haplotypes
    diploid_ne                       pi_branch / (4 MUT_RATE): tsdate's population_size and SINGER's -Ne, in
                                     diploid individuals
    trees_md5                        the md5 of the .trees file, which changes with every run, since the
                                     provenance of the file holds the time it was written

Run with (the Snakefile runs one model and size per job):
  GNN_VS_DIR=training_data/const GNN_VS_SCENARIOS=const GNN_VS_SIZES=10 GNN_VS_FILES=24 \
    GNN_VS_SEQLEN=20_000_000 python -u src/simulate_data.py training_data
The argument is the data set, training_data or testing_data, which sets the seed bases.
Env: GNN_VS_DIR and GNN_VS_SEQLEN (both required), GNN_VS_SCENARIOS and GNN_VS_SIZES (comma lists, required), and
GNN_VS_FILES (default 24), and for papuans_10j19 GNN_PAP_NEA, GNN_PAP_DEN, and GNN_PAP_YRI (defaults 2, 2, 24).
"""

import csv
import hashlib
import os
import sys

import msprime
import numpy as np
import stdpopsim

OUT_DIR    = os.path.abspath(os.environ.get("GNN_VS_DIR", ""))
SEQ_LENGTH = int(os.environ.get("GNN_VS_SEQLEN", "0"))
SCENARIOS = [s for s in os.environ.get("GNN_VS_SCENARIOS", "").split(",") if s]
SIZES     = [int(x) for x in os.environ.get("GNN_VS_SIZES", "").split(",") if x]
N_FILES   = int(os.environ.get("GNN_VS_FILES", "24"))
# Haplotypes of NeaA, DenA, and YRI in papuans_10j19, as in generate_varsize_data.py, with the panel of the
# real-data run as defaults. The other haplotypes of a size are Papuan.
PAP_N_NEA = int(os.environ.get("GNN_PAP_NEA", "2"))
PAP_N_DEN = int(os.environ.get("GNN_PAP_DEN", "2"))
PAP_N_YRI = int(os.environ.get("GNN_PAP_YRI", "24"))
# Rates per base pair per generation, the size of const in diploid individuals, and the fewest sites of a
# replicate, as in generate_varsize_data.py.
RECOMB_RATE = 1e-8
MUT_RATE    = 1.25e-8
NE          = 10_000
MIN_SITES   = 200
# Every model this script knows, in a fixed order: its place sets its seeds.
MODELS = ["const", "split", "bott", "ooa", "zigzag_1s14", "papuans_10j19"]
SEED_BASES = {("training_data", "ancestry"): 100_000_000, ("training_data", "mutation"): 200_000_000,
              ("testing_data", "ancestry"): 300_000_000, ("testing_data", "mutation"): 400_000_000,
              ("testing_data", "tool"): 500_000_000}
# Largest allowed ratio between the mean pairwise coalescence time of the true trees, averaged over replicates,
# and msprime's expectation. ploidy=1 gives expected ratios between 0.42 (bott) and 0.66 (ooa).
TIME_SCALE_TOLERANCE = 1.25


def stdpopsim_demography(model_id):
    """The msprime demography of one HomSap model of stdpopsim."""
    return stdpopsim.get_species("HomSap").get_demographic_model(model_id).model


def model_demography(model):
    """The msprime demography of one model, whose sizes count diploid individuals: const, split, and bott as in
    generate_varsize_data.py (simulate_const, simulate_split, simulate_bott), and ooa, zigzag_1s14, and
    papuans_10j19 from stdpopsim."""
    if model == "const":
        return msprime.Demography.isolated_model([NE])
    if model == "split":
        demography = msprime.Demography()
        demography.add_population(name="A", initial_size=20_000)
        demography.add_population(name="B", initial_size=5_000)
        demography.add_population(name="ANC", initial_size=NE)
        demography.add_population_split(time=10_000, derived=["A", "B"], ancestral="ANC")
        return demography
    if model == "bott":
        demography = msprime.Demography()
        demography.add_population(name="A", initial_size=NE)
        demography.add_population_parameters_change(time=500, population="A", initial_size=500)
        demography.add_population_parameters_change(time=700, population="A", initial_size=NE)
        return demography
    if model == "ooa":
        return stdpopsim_demography("OutOfAfricaArchaicAdmixture_5R19")
    if model == "zigzag_1s14":
        return stdpopsim_demography("Zigzag_1S14")
    if model == "papuans_10j19":
        return stdpopsim_demography("PapuansOutOfAfrica_10J19")
    raise ValueError(f"unknown model {model!r} (known: {', '.join(MODELS)})")


def sampling_scheme(model, size):
    """Number of haplotypes sampled from each population of one model, as generate_varsize_data.py samples
    them."""
    if model in ("const", "zigzag_1s14"):
        return {"pop_0" if model == "const" else "generic": size}
    if model == "split":
        return {"A": size // 2, "B": size - size // 2}
    if model == "bott":
        return {"A": size}
    if model == "ooa":
        return {"YRI": size // 3, "CEU": size // 3, "CHB": size - 2 * (size // 3)}
    if model == "papuans_10j19":
        papuan = size - PAP_N_NEA - PAP_N_DEN - PAP_N_YRI
        if papuan < 1:
            raise ValueError(f"papuans_10j19 at {size} haplotypes leaves {papuan} Papuan haplotypes")
        return {"Papuan": papuan, "YRI": PAP_N_YRI, "NeaA": PAP_N_NEA, "DenA": PAP_N_DEN}
    raise ValueError(f"unknown model {model!r} (known: {', '.join(MODELS)})")


def mutate(ts, seed):
    """Mutations at MUT_RATE, without the sites that more than one mutation hit, as generate_varsize_data._mutate
    adds them, and the number of sites removed."""
    mts = msprime.sim_mutations(ts, rate=MUT_RATE, random_seed=seed)
    # msprime's genome is discrete, so two mutations can hit one base pair.
    removed = [site.id for site in mts.sites() if len(site.mutations) > 1]
    return mts.delete_sites(removed), len(removed)


def simulate(model, size, ancestry_seed, mutation_seed):
    """One true tree sequence with its mutations (ploidy=2 with one haplotype per sample, then mutate), and the
    number of sites that mutate removed."""
    samples = [msprime.SampleSet(count, population=population, ploidy=1)
               for population, count in sampling_scheme(model, size).items()]
    ts = msprime.sim_ancestry(samples=samples, demography=model_demography(model), ploidy=2,
                              recombination_rate=RECOMB_RATE, sequence_length=SEQ_LENGTH,
                              random_seed=ancestry_seed)
    return mutate(ts, mutation_seed)


def check_simulation(mts, model, size):
    """The failed checks of one true tree sequence, as strings (an empty list if it passes)."""
    problems = []
    samples = mts.samples()
    # Haplotypes per population, against the sampling scheme.
    names = [mts.population(mts.node(node).population).metadata["name"] for node in samples]
    counts = {name: names.count(name) for name in set(names)}
    if len(samples) != size or counts != sampling_scheme(model, size):
        problems.append(f"samples per population {counts}, expected {sampling_scheme(model, size)}")
    # Each haplotype is at the sampling time of its population, which msprime leaves at 0 when the model sets none.
    demography = model_demography(model)
    sampling_times = [demography[name].default_sampling_time or 0 for name in names]
    if np.any(mts.nodes_time[samples] != sampling_times):
        problems.append("a sample is not at the sampling time of its population")
    if mts.sequence_length != SEQ_LENGTH:
        problems.append(f"sequence length {mts.sequence_length}, expected {SEQ_LENGTH}")
    mutations_per_site = np.bincount(mts.mutations_site, minlength=mts.num_sites)
    if np.any(mutations_per_site != 1):
        problems.append(f"{int(np.sum(mutations_per_site != 1))} sites without exactly one mutation")
    if mts.num_sites < MIN_SITES:
        problems.append(f"{mts.num_sites} sites, fewer than MIN_SITES = {MIN_SITES}")
    return problems


def md5_of_file(path):
    """The md5 checksum of a file, read one megabyte at a time."""
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed(data_set, purpose, model, size, replicate):
    """The seed of one replicate of a data set for one purpose (ancestry, mutation, or tool)."""
    return SEED_BASES[(data_set, purpose)] + 1_000_000 * MODELS.index(model) + 1_000 * size + replicate


def simulate_model_size(data_set, model, size):
    """Simulate and write every replicate of one model and size of a data set, write their table, and return
    the failed checks."""
    t2_msprime = float(model_demography(model).debug().mean_coalescence_time(sampling_scheme(model, size)))
    rows, problems = [], []
    for replicate in range(N_FILES):
        tag = f"{model}_S{size}_{replicate}"
        seeds = {purpose: seed(data_set, purpose, model, size, replicate)
                 for (seed_data_set, purpose) in SEED_BASES if seed_data_set == data_set}
        mts, removed_var_sites = simulate(model, size, seeds["ancestry"], seeds["mutation"])
        path = os.path.join(OUT_DIR, f"{tag}.trees")
        mts.dump(path)
        branch_diversity = float(mts.diversity(mode="branch"))
        # The mean pairwise coalescence time is half the branch diversity, and pi_branch is mu times the branch
        # diversity, the diversity that the true trees predict.
        pi_branch = MUT_RATE * branch_diversity
        row = {"model": model, "size": size, "replicate": replicate}
        row |= {f"{purpose}_seed": value for purpose, value in seeds.items()}
        row |= {"sequence_length": int(mts.sequence_length), "removed_var_sites": removed_var_sites,
                "qced_var_sites": mts.num_sites, "trees": mts.num_trees,
                "pi_site": float(mts.diversity(mode="site")), "pi_branch": pi_branch,
                "t2_branch": branch_diversity / 2, "t2_msprime": t2_msprime,
                "relate_ne": int(round(pi_branch / (2 * MUT_RATE))), "diploid_ne": pi_branch / (4 * MUT_RATE),
                "trees_md5": md5_of_file(path)}
        rows.append(row)
        problems += [f"{tag}: {problem}" for problem in check_simulation(mts, model, size)]
        print(f"{tag}: seeds {', '.join(str(value) for value in seeds.values())}, {mts.num_sites} sites kept, "
              f"{removed_var_sites} removed, {mts.num_trees} trees, t2_branch / t2_msprime "
              f"{row['t2_branch'] / t2_msprime:.3f}", flush=True)
    mean_ratio = float(np.mean([row["t2_branch"] / t2_msprime for row in rows]))
    print(f"{model} S{size}: mean t2_branch / t2_msprime over {N_FILES} replicates {mean_ratio:.3f} "
          f"(t2_msprime {t2_msprime:.1f} generations)", flush=True)
    if not 1 / TIME_SCALE_TOLERANCE <= mean_ratio <= TIME_SCALE_TOLERANCE:
        problems.append(f"{model} S{size}: mean t2_branch / t2_msprime {mean_ratio:.3f}")
    seeds_used = [row[key] for row in rows for key in row if key.endswith("_seed")]
    if len(set(seeds_used)) != len(seeds_used):
        problems.append(f"{model} S{size}: a seed is used twice")
    with open(os.path.join(OUT_DIR, f"{model}_S{size}_simulations.csv"), "w", newline="") as fh:
        # Unix line ends, so that shell tools read the last column without a carriage return.
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return problems


def main():
    """Simulate every model and size of GNN_VS_SCENARIOS and GNN_VS_SIZES for the data set given as the
    argument, then end with an error if a check failed."""
    if len(sys.argv) != 2 or sys.argv[1] not in ("training_data", "testing_data"):
        raise SystemExit("usage: python simulate_data.py training_data|testing_data")
    data_set = sys.argv[1]
    unknown = [model for model in SCENARIOS if model not in MODELS]
    if unknown or not SCENARIOS or not SIZES:
        raise SystemExit(f"!! GNN_VS_SCENARIOS {SCENARIOS} and GNN_VS_SIZES {SIZES} must be set, with models "
                         f"among {MODELS}")
    # The folder and the length have no default, so that a job never writes data of the wrong length or place.
    if not os.environ.get("GNN_VS_DIR") or SEQ_LENGTH <= 0:
        raise SystemExit("!! GNN_VS_DIR and GNN_VS_SEQLEN must be set")
    os.makedirs(OUT_DIR, exist_ok=True)
    # The seed formula keeps seeds distinct only for sizes and replicates below 1,000, and a table needs a row.
    if max(SIZES) >= 1_000 or not 1 <= N_FILES <= 1_000:
        raise SystemExit(f"!! sizes {SIZES} must be below 1,000 and GNN_VS_FILES {N_FILES} from 1 to 1,000")
    print(f"simulate_data | {data_set} | models {SCENARIOS} | sizes {SIZES} | {N_FILES} replicates | "
          f"{SEQ_LENGTH:,} bp | {OUT_DIR}", flush=True)
    problems = []
    for model in SCENARIOS:
        for size in SIZES:
            problems += simulate_model_size(data_set, model, size)
    if problems:
        raise SystemExit("!! failed checks:\n  " + "\n  ".join(problems))
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
