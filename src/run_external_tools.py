"""
run_external_tools.py: the external tools of the benchmark on the testing data
===============================================================================
Runs one external tool on testing files of src/simulate_data.py with the settings of the constants below, and keeps
the files that the scoring reads. The script
needs no other script of src/. Every tool gets the rates of the simulation (MUT_RATE and RECOMB_RATE, per base pair per
generation), the effective population size from the branch diversity of the true trees (relate_ne or diploid_ne of the
table of simulate_data.py), and the tool seed of the file (tool_seed of the same table). No tool gets a history of
population sizes.

Tools:
  relate         Relate 1.2.4: --mode All -m MUT_RATE -N relate_ne, with a genetic map of RECOMB_RATE (1 cM/Mb), then
                 EstimatePopulationSize.sh --threshold 0 --num_iter 5, which re-dates every tree with the coalescence
                 rates that it estimates, then SampleBranchLengths.sh --num_samples 100 --format n on the re-dated
                 trees, with Relate's default number of proposals. --threshold is a fraction of the trees, left out of
                 the estimate of the rates, and Relate's default is 0.5.
  tsinfer        tsinfer 0.5.1 with recombination_rate=RECOMB_RATE and mismatch_ratio=0, then tsdate 0.2.7:
                 preprocess_ts(erase_flanks=False), then date(method="variational_gamma",
                 match_segregating_sites=True).
  true_topology  The simulated topology as argGNN gives its topologies to the dating: the nodes and edges of the
                 true tree sequence, each node at a placeholder time (its height, the most edges on a path down to
                 a sample), and the mutations placed by parsimony from the genotype matrix, as
                 branch_length._assemble places those of argGNN. Then, separately, the same tsdate.date call and
                 POLEGON with the settings given for singer below, each on that tree sequence, so POLEGON starts
                 from the placeholder times.
  singer         SINGER 0.1.9: -Ne diploid_ne -m MUT_RATE -ratio RECOMB_RATE / MUT_RATE -polar 0.99 -ploidy 1 -n 600
                 -thin 20. Sample k ends at iteration 20 (k + 1), so samples 0 to 499 (the first 10,000 iterations)
                 are the burn-in, and samples 500 to 599 (iterations 10,020 to 12,000) are the posterior samples. Then
                 POLEGON 0.1.3 on each posterior sample, from SINGER's node times: -burn_in 100 -num_samples 100
                 -thin 10 -scaling_rep 5. POLEGON takes no seed.

Inputs: testing files, testing_data/{model}/{model}_S{size}_{replicate}.trees, and the table that simulate_data.py
writes next to them, {model}_S{size}_simulations.csv. The md5 of each file must equal trees_md5 of its row.

Outputs, in OUT_DIR/{model}_S{size}_{replicate}/ for each file. The folder is emptied first, and it holds exactly
these files at the end:
  relate         relate.trees (the re-dated trees), relate.coal (the coalescence rates of EstimatePopulationSize),
                 relate_sample_branch_lengths.newick.gz and relate_sample_branch_lengths.sites (the 100 draws)
  tsinfer        tsinfer.trees (dated) and tsinfer_node_posteriors.npz (the mean and the variance of the gamma
                 posterior of each node, in node order)
  true_topology  true_topology.trees and true_topology_node_posteriors.npz, as for tsinfer, and
                 true_topology_polegon.npz (mean, the posterior-mean age of each node in node order, and draws, the
                 100 draws of POLEGON, one row per node and one column per draw)
  singer         singer_{k}.trees for k = 500 to 599, singer.log (the log of SINGER), singer_trace.csv (one row per
                 sample of the chain: sample, iteration, incompatible_sites, diversity_mse), and singer_polegon.npz
                 (for each k, mean_{k}, the posterior-mean age of each node of singer_{k}.trees in node order, and
                 last_draw_{k}, the last draw of POLEGON)
  every tool     run_times.csv: the wall time in seconds of each step, and of the whole data set as "task"
In the trace, incompatible_sites is the number of mutations that SINGER could not map to one branch, from the
"rethread" rows of singer.log (-1 for a sample without a row), and diversity_mse is the mean over the 10 kb windows
between the first and the last site of (MUT_RATE times the branch diversity of the sample minus the site diversity of
the data) squared. The printed lines, which the Snakefile keeps as
the log of each job, hold the versions, the md5 of each binary, the seed, the Ne, and every command, with the paths
inside the repo written relative to the repo root. tsdate and tsinfer are imported only by the tools that use them,
since the import of tsdate takes about a minute.

Run with (the Snakefile runs one file per job for relate and singer, and one model and size per job for tsinfer and
true_topology):
  python -u src/run_external_tools.py TOOL OUT_DIR TRUE_FILE [TRUE_FILE ...]
Env: GNN_RELATE_BIN, GNN_RELATE_FMT, GNN_RELATE_SCRIPTS, GNN_SINGER_DIR, and GNN_POLEGON_DIR, absolute paths, by default
the tools in bin/ of the repo.
"""

import contextlib
import csv
import glob
import gzip
import hashlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import tskit

# The tools, by default in bin/ of the repo, as on Oscar. Relate runs in a temporary folder, so the paths are absolute.
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_DIR = os.path.join(REPO_DIR, "bin")
RELATE_BIN = os.environ.get("GNN_RELATE_BIN", os.path.join(BIN_DIR, "relate_v1.2.4_x86_64_static", "bin", "Relate"))
RELATE_FMT = os.environ.get("GNN_RELATE_FMT",
                            os.path.join(BIN_DIR, "relate_v1.2.4_x86_64_static", "bin", "RelateFileFormats"))
RELATE_SCRIPTS = os.environ.get("GNN_RELATE_SCRIPTS", os.path.join(BIN_DIR, "relate_v1.2.4_x86_64_static", "scripts"))
SINGER_DIR = os.environ.get("GNN_SINGER_DIR", os.path.join(BIN_DIR, "singer-0.1.9-beta-linux-x86_64"))
POLEGON_DIR = os.environ.get("GNN_POLEGON_DIR", os.path.join(BIN_DIR, "polegon-0.1.3-alpha-linux-x86_64"))
# Rates per base pair per generation, those of simulate_data.py.
RECOMB_RATE = 1e-8
MUT_RATE    = 1.25e-8
# Relate: EstimatePopulationSize's threshold and iterations, and the number of draws of SampleBranchLengths.
RELATE_THRESHOLD  = "0"
RELATE_ITERATIONS = 5
RELATE_DRAWS      = 100
# SINGER: samples of the chain, iterations between two samples, and samples of the burn-in.
SINGER_SAMPLES = 600
SINGER_THIN    = 20
SINGER_BURN_IN = 500
SINGER_KEPT    = list(range(SINGER_BURN_IN, SINGER_SAMPLES))
# convert_to_tskit converts this many samples per call, so that the burn-in samples never all exist as .trees.
SINGER_CONVERT_BLOCK = 50
# POLEGON's defaults, except five rescaling steps, as in its paper (deng2025general, Appendix 1A), in place of three.
POLEGON_SETTINGS = {"burn_in": 100, "num_samples": 100, "thin": 10, "scaling_rep": 5}
# Windows of the diversity fit in the trace, in base pairs, the WINDOW_SIZE of arg_metrics.py.
WINDOW_SIZE = 10_000
TOOLS = ("relate", "tsinfer", "true_topology", "singer")
# The files of each data set at the end of a run. The Snakefile lists the same names as the outputs of each rule.
OUTPUT_FILES = {
    "relate": ["relate.trees", "relate.coal", "relate_sample_branch_lengths.newick.gz",
               "relate_sample_branch_lengths.sites", "run_times.csv"],
    "tsinfer": ["tsinfer.trees", "tsinfer_node_posteriors.npz", "run_times.csv"],
    "true_topology": ["true_topology.trees", "true_topology_node_posteriors.npz", "true_topology_polegon.npz",
                      "run_times.csv"],
    "singer": [f"singer_{k}.trees" for k in SINGER_KEPT]
              + ["singer.log", "singer_trace.csv", "singer_polegon.npz", "run_times.csv"],
}


@contextlib.contextmanager
def timed_step(step, steps):
    """Append the wall time of the enclosed block to steps, as one row named step, and print it."""
    start = time.perf_counter()
    yield
    steps.append({"step": step, "wall_seconds": round(time.perf_counter() - start, 2)})
    print(f"  {step}: {steps[-1]['wall_seconds']} s", flush=True)


def md5_of_file(path):
    """The md5 checksum of a file, read one megabyte at a time."""
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    """Write a list of dictionaries as a table with Unix line ends."""
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_simulation(path):
    """One testing file, its row of the table of simulate_data.py, and its name {model}_S{size}_{replicate}, after a
    check that the md5 of the file equals trees_md5 of the row."""
    tag = os.path.basename(path)[:-len(".trees")]
    stem, replicate = tag.rsplit("_", 1)
    table = os.path.join(os.path.dirname(os.path.abspath(path)), f"{stem}_simulations.csv")
    if not os.path.exists(table):
        raise SystemExit(f"!! no table {relative(table)}")
    with open(table, newline="") as fh:
        rows = {row["replicate"]: row for row in csv.DictReader(fh)}
    if replicate not in rows:
        raise SystemExit(f"!! {relative(table)} has no row for replicate {replicate}")
    if md5_of_file(path) != rows[replicate]["trees_md5"]:
        raise SystemExit(f"!! {relative(os.path.abspath(path))} is not the file that {relative(table)} lists (its md5 "
                         f"differs)")
    return tskit.load(path), rows[replicate], tag


def relative(text):
    """text with every path inside the repo written relative to the repo root, so that a log holds no home path."""
    return text.replace(REPO_DIR + os.sep, "")


def run_command(command, cwd):
    """Print a command, run it in cwd, and end the script with its last output lines if it fails."""
    print("  $ " + relative(" ".join(str(word) for word in command)), flush=True)
    result = subprocess.run([str(word) for word in command], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"!! exit status {result.returncode}: {relative(result.stderr or result.stdout)[-600:]}")
    return result


def write_vcf(mts, path):
    """The haploid VCF of the data, with a site at position 0 moved to 1, since tskit writes no POS of 0."""
    with open(path, "w") as fh:
        mts.write_vcf(fh, position_transform=lambda x: np.fmax(1, x))


def write_genetic_map(path, sequence_length):
    """A genetic map of RECOMB_RATE in Relate's format, as generate_varsize_data.write_genetic_map writes it: 1,000
    positions from 0 to the sequence length, the rate in cM/Mb, and the map position in cM."""
    positions = np.linspace(0, sequence_length, 1000)
    with open(path, "w") as fh:
        fh.write("pos COMBINED_rate Genetic_Map\n")
        for position, centimorgans in zip(positions, positions * RECOMB_RATE * 100):
            fh.write(f"{int(position)} {RECOMB_RATE * 1e8:.6f} {centimorgans:.6f}\n")


def write_poplabels(sample_file, path):
    """A poplabels file with one population for EstimatePopulationSize, from the sample IDs of Relate's .sample file,
    as generate_varsize_data._write_poplabels writes it."""
    with open(sample_file) as fh:
        ids = [line.split()[0] for line in fh.read().splitlines()[2:] if line.split()]
    with open(path, "w") as fh:
        fh.write("sample population group sex\n")
        for sample_id in ids:
            fh.write(f"{sample_id} POP GRP 1\n")


def run_relate(mts, row, folder, steps):
    """Relate's --mode All, EstimatePopulationSize.sh, and SampleBranchLengths.sh on one testing file. Relate runs in a
    temporary folder on the node, since on Oscar's scratch file system it cannot delete its own temporary folders, and
    only the output files are copied to folder."""
    seed = row["tool_seed"]
    run_dir = tempfile.mkdtemp(prefix="relate_")
    try:
        write_vcf(mts, os.path.join(run_dir, "data.vcf"))
        write_genetic_map(os.path.join(run_dir, "genetic_map.txt"), mts.sequence_length)
        with timed_step("relate_mode_all", steps):
            run_command([RELATE_FMT, "--mode", "ConvertFromVcf", "--haps", "data.haps", "--sample", "data.sample",
                         "--input", "data"], run_dir)
            run_command([RELATE_BIN, "--mode", "All", "-m", MUT_RATE, "-N", row["relate_ne"], "--haps", "data.haps",
                         "--sample", "data.sample", "--map", "genetic_map.txt", "--seed", seed, "-o", "relate"],
                        run_dir)
        write_poplabels(os.path.join(run_dir, "data.sample"), os.path.join(run_dir, "data.poplabels"))
        with timed_step("relate_estimate_population_size", steps):
            run_command(["bash", os.path.join(RELATE_SCRIPTS, "EstimatePopulationSize", "EstimatePopulationSize.sh"),
                         "-i", "relate", "-o", "relate_redated", "-m", MUT_RATE, "--poplabels", "data.poplabels",
                         "--num_iter", RELATE_ITERATIONS, "--threshold", RELATE_THRESHOLD, "--seed", seed, "--noplot"],
                        run_dir)
            run_command([RELATE_FMT, "--mode", "ConvertToTreeSequence", "-i", "relate_redated", "-o",
                         "relate_redated"], run_dir)
        with timed_step("relate_sample_branch_lengths", steps):
            run_command(["bash", os.path.join(RELATE_SCRIPTS, "SampleBranchLengths", "SampleBranchLengths.sh"),
                         "-i", "relate_redated", "-o", "relate_sample_branch_lengths", "-m", MUT_RATE, "--coal",
                         "relate_redated.coal", "--format", "n", "--num_samples", RELATE_DRAWS, "--seed", seed],
                        run_dir)
        # The newick file has one line per tree and draw, so it is gzipped.
        with open(os.path.join(run_dir, "relate_sample_branch_lengths.newick"), "rb") as source, \
                gzip.open(os.path.join(folder, "relate_sample_branch_lengths.newick.gz"), "wb") as target:
            shutil.copyfileobj(source, target)
        for source, target in (("relate_redated.trees", "relate.trees"), ("relate_redated.coal", "relate.coal"),
                               ("relate_sample_branch_lengths.sites", "relate_sample_branch_lengths.sites")):
            shutil.copy(os.path.join(run_dir, source), os.path.join(folder, target))
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def date_with_tsdate(ts, name, folder, steps):
    """tsdate's variational_gamma with match_segregating_sites=True on one topology, kept as {name}.trees with the
    mean and the variance of the gamma posterior of each node in {name}_node_posteriors.npz. The node times of the
    tree sequence are tsdate's posterior means, raised where needed so that each parent is older than its children,
    so they can differ from the saved means."""
    import tsdate
    with timed_step("tsdate", steps):
        dated, fit = tsdate.date(ts, mutation_rate=MUT_RATE, method="variational_gamma",
                                 match_segregating_sites=True, return_fit=True)
    dated.dump(os.path.join(folder, f"{name}.trees"))
    posteriors = fit.node_posteriors()
    np.savez(os.path.join(folder, f"{name}_node_posteriors.npz"), mean=posteriors["mean"],
             variance=posteriors["variance"])


def run_tsinfer(mts, row, folder, steps):
    """tsinfer on the data of one testing file, then preprocess_ts and tsdate. The sample data are the biallelic
    sites with 0 as the ancestral allele, without monomorphic sites, as generate_varsize_data.run_tsinfer builds
    them."""
    import tsdate
    import tsinfer
    with tsinfer.SampleData(sequence_length=mts.sequence_length) as sample_data:
        for variant in mts.variants():
            genotypes = (np.asarray(variant.genotypes) != 0).astype(np.int8)
            if genotypes.min() == genotypes.max():
                continue
            sample_data.add_site(variant.site.position, genotypes, ["0", "1"], ancestral_allele=0)
    print(f"  tsinfer.infer(recombination_rate={RECOMB_RATE}, mismatch_ratio=0); "
          f"tsdate.preprocess_ts(erase_flanks=False); tsdate.date(mutation_rate={MUT_RATE}, "
          f"method='variational_gamma', match_segregating_sites=True)", flush=True)
    with timed_step("tsinfer", steps):
        inferred = tsinfer.infer(sample_data, recombination_rate=RECOMB_RATE, mismatch_ratio=0)
    # With mismatch_ratio=0 every site should carry exactly one mutation, as in the data.
    mutations_per_site = np.bincount(inferred.mutations_site, minlength=inferred.num_sites)
    if np.any(mutations_per_site != 1):
        raise SystemExit(f"!! tsinfer: {int(np.sum(mutations_per_site != 1))} sites without exactly one mutation")
    with timed_step("preprocess_ts", steps):
        preprocessed = tsdate.preprocess_ts(inferred, erase_flanks=False)
    print(f"  tsinfer: {inferred.num_nodes} nodes, {preprocessed.num_nodes} after preprocess_ts", flush=True)
    date_with_tsdate(preprocessed, "tsinfer", folder, steps)


def simulated_topology(mts):
    """The simulated topology of one testing file as argGNN gives its topologies to the dating: the nodes and edges of
    the true tree sequence, each node at a placeholder time, and the mutations placed again by parsimony. The
    placeholder of a node is its height, the most edges on a path down to a sample, so a parent is always older than
    its children and no simulated time enters as a value. The node IDs of the simulation are kept, so the edges of
    one height stay in the order of their simulated times, the order of tsdate's updates. A random order of the nodes
    changes the dates little. The parsimony is that of branch_length._assemble: at each site with both alleles,
    Tree.map_mutations with the alleles 0 and 1 gives the ancestral state and the mutations, and the mutation parents
    are computed again."""
    # The height of each node. The edges are taken from the youngest child up, so the height of a child is final
    # before its parents use it.
    height = np.zeros(mts.num_nodes)
    for edge in np.argsort(mts.nodes_time[mts.edges_child], kind="stable"):
        parent, child = mts.edges_parent[edge], mts.edges_child[edge]
        height[parent] = max(height[parent], height[child] + 1)
    tables = mts.dump_tables()
    tables.nodes.time = height
    tables.sites.clear()
    tables.mutations.clear()
    tables.sort()
    topology = tables.tree_sequence()
    tree = topology.first()
    for position, genotypes in zip(mts.tables.sites.position, mts.genotype_matrix()):
        if genotypes.min() == genotypes.max():
            continue
        tree.seek(position)
        ancestral_state, mutations = tree.map_mutations(genotypes, ["0", "1"])
        site = tables.sites.add_row(position=position, ancestral_state=ancestral_state)
        for mutation in mutations:
            tables.mutations.add_row(site=site, node=mutation.node, derived_state=mutation.derived_state)
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()


def run_true_topology(mts, row, folder, steps):
    """The two references on the simulated topology of one testing file: tsdate and, separately, POLEGON, each on
    simulated_topology(mts). POLEGON starts from the placeholder times, as it would from those of argGNN, and its
    posterior means and all its draws are kept. The printed line counts the sites whose mutations differ in node
    from the one mutation of the simulation."""
    with timed_step("parsimony", steps):
        topology = simulated_topology(mts)
    simulated_nodes = dict(zip(mts.tables.sites.position[mts.mutations_site].tolist(), mts.mutations_node.tolist()))
    nodes = {}
    for position, node in zip(topology.tables.sites.position[topology.mutations_site], topology.mutations_node):
        nodes.setdefault(float(position), []).append(int(node))
    moved = sum(nodes.get(position, []) != [node] for position, node in simulated_nodes.items())
    print(f"  simulated_topology: {topology.num_nodes} nodes at heights up to {topology.nodes_time.max():.0f}, "
          f"{topology.num_mutations} mutations at {topology.num_sites} sites, {moved} sites whose mutations differ "
          f"from the simulation", flush=True)
    print(f"  tsdate.date(mutation_rate={MUT_RATE}, method='variational_gamma', match_segregating_sites=True)",
          flush=True)
    date_with_tsdate(topology, "true_topology", folder, steps)
    print(f"  POLEGON, as in.trees of a temporary folder: $ {relative(' '.join(polegon_command()))}", flush=True)
    with timed_step("polegon", steps):
        mean, draws = polegon_ages(topology, folder)
    np.savez(os.path.join(folder, "true_topology_polegon.npz"), mean=mean, draws=draws.astype(np.float32))


def window_edges(ts, starts):
    """tskit window edges from 0 to the sequence length whose inner windows are the 10 kb windows that start at
    starts, and the indices of those inner windows, as arg_metrics.tskit_windows builds them."""
    inner_edges = np.append(starts, starts[-1] + WINDOW_SIZE)
    edges = np.unique(np.concatenate([[0.0], inner_edges, [ts.sequence_length]]))
    first = int(np.searchsorted(edges, inner_edges[0]))
    return edges, np.arange(first, first + len(starts))


def read_incompatible(log_path):
    """The number of mutations that SINGER could not map to one branch, per sample, from the 'rethread' rows of its
    log. A resumed run logs a sample again, and the last row is kept."""
    by_sample = {}
    with open(log_path) as fh:
        next(fh)
        for line in fh:
            words = line.rstrip("\n").split("\t")
            if len(words) >= 5 and words[2] == "rethread":
                by_sample[int(words[1])] = int(words[4])
    return by_sample


def polegon_command():
    """The command of polegon_master on in.trees of its folder, with the settings of POLEGON_SETTINGS."""
    return [sys.executable, os.path.join(POLEGON_DIR, "polegon_master"), "-m", str(MUT_RATE), "-input", "in",
            "-output", "out", "-burn_in", str(POLEGON_SETTINGS["burn_in"]),
            "-num_samples", str(POLEGON_SETTINGS["num_samples"]), "-thin", str(POLEGON_SETTINGS["thin"]),
            "-scaling_rep", str(POLEGON_SETTINGS["scaling_rep"]), "-write_samples"]


def polegon_ages(input_ts, folder):
    """POLEGON's posterior-mean age of each node of input_ts and its draws (one row per node, one column per draw),
    from the node times of input_ts. polegon_master runs in a temporary folder inside folder. The ages come from
    {input}_new_nodes.txt, not from the tree sequence that polegon_master rebuilds, which tskit 1.0 rejects at a site
    with nested mutations (BUGS.md). That rebuild comes after the
    binary has run, so its failure gives polegon_master an exit status of 1, and the run counts as done when
    polegon_master reports that the binary succeeded. POLEGON rescales the posterior mean and each draw in the same
    way."""
    run_dir = tempfile.mkdtemp(prefix="polegon_", dir=folder)
    try:
        input_ts.dump(os.path.join(run_dir, "in.trees"))
        result = subprocess.run(polegon_command(), capture_output=True, text=True, cwd=run_dir)
        if "polegon executed successfully" not in result.stdout:
            raise SystemExit(f"!! POLEGON failed (exit status {result.returncode}): "
                             f"{relative(result.stderr or result.stdout)[-600:]}")
        mean = np.loadtxt(os.path.join(run_dir, "in_new_nodes.txt"), ndmin=1)
        # One row per node and one column per draw.
        draws = np.loadtxt(os.path.join(run_dir, "out_node_samples.txt"), ndmin=2)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    if mean.shape[0] != input_ts.num_nodes or draws.shape[0] != input_ts.num_nodes:
        raise SystemExit(f"!! POLEGON: {mean.shape[0]} ages and {draws.shape[0]} rows of draws for "
                         f"{input_ts.num_nodes} nodes")
    return mean, draws


def run_singer(mts, row, folder, steps):
    """One SINGER chain on one testing file, its trace, and POLEGON on each posterior sample. SINGER runs in folder.
    convert_to_tskit converts the samples in blocks of SINGER_CONVERT_BLOCK, each sample gives its row of the trace,
    and the .trees of the burn-in samples are deleted. Then every text file of SINGER and the VCF are deleted."""
    write_vcf(mts, os.path.join(folder, "data.vcf"))
    command = [sys.executable, os.path.join(SINGER_DIR, "singer_master"), "-Ne", row["diploid_ne"], "-m", MUT_RATE,
               "-ratio", RECOMB_RATE / MUT_RATE, "-vcf", "data", "-output", "singer", "-start", 0,
               "-end", int(mts.sequence_length), "-n", SINGER_SAMPLES, "-thin", SINGER_THIN, "-polar", 0.99,
               "-ploidy", 1, "-seed", row["tool_seed"]]
    with timed_step("singer_chain", steps):
        result = run_command(command, folder)
    # SINGER prints about 12 lines per iteration, so only the command of each start of the chain is printed. After a
    # failure, singer_master starts the chain again with -debug and another seed, and says so.
    for line in result.stdout.splitlines():
        if " -seed " in line or "Auto-debug" in line:
            print("  " + relative(line), flush=True)
    # The windows of the diversity fit lie between the first and the last site of the data.
    positions = mts.tables.sites.position
    starts = np.arange(np.ceil(positions[0] / WINDOW_SIZE) * WINDOW_SIZE, positions[-1] // WINDOW_SIZE * WINDOW_SIZE,
                       WINDOW_SIZE)
    edges, inner = window_edges(mts, starts)
    data_diversity = mts.diversity(mode="site", windows=edges)[inner]
    incompatible = read_incompatible(os.path.join(folder, "singer.log"))
    trace = []
    with timed_step("singer_convert_and_trace", steps):
        for first in range(0, SINGER_SAMPLES, SINGER_CONVERT_BLOCK):
            last = min(first + SINGER_CONVERT_BLOCK, SINGER_SAMPLES)
            # The -end of convert_to_tskit is exclusive.
            run_command([sys.executable, os.path.join(SINGER_DIR, "convert_to_tskit"), "-input", "singer", "-output",
                         "singer", "-start", first, "-end", last, "-step", 1], folder)
            for k in range(first, last):
                path = os.path.join(folder, f"singer_{k}.trees")
                sample_ts = tskit.load(path)
                edges, inner = window_edges(sample_ts, starts)
                diversity = MUT_RATE * sample_ts.diversity(mode="branch", windows=edges)[inner]
                trace.append({"sample": k, "iteration": SINGER_THIN * (k + 1),
                              "incompatible_sites": incompatible.get(k, -1),
                              "diversity_mse": float(np.mean(np.square(diversity - data_diversity)))})
                if k < SINGER_BURN_IN:
                    os.remove(path)
    write_csv(os.path.join(folder, "singer_trace.csv"), trace)
    # Every text file of SINGER (the nodes, branches, mutations, and recombinations of each sample), and the VCF.
    for path in glob.glob(os.path.join(folder, "singer_*.txt")) + [os.path.join(folder, "data.vcf")]:
        os.remove(path)
    polegon = {}
    print(f"  POLEGON on each posterior sample, as in.trees of a temporary folder: "
          f"$ {relative(' '.join(polegon_command()))}", flush=True)
    with timed_step("polegon", steps):
        for k in SINGER_KEPT:
            polegon[f"mean_{k}"], draws = polegon_ages(tskit.load(os.path.join(folder, f"singer_{k}.trees")), folder)
            polegon[f"last_draw_{k}"] = draws[:, -1].astype(np.float32)
    np.savez(os.path.join(folder, "singer_polegon.npz"), **polegon)


RUN = {"relate": run_relate, "tsinfer": run_tsinfer, "true_topology": run_true_topology, "singer": run_singer}


def print_versions(tool):
    """Print the versions of Python and of the packages, and the path and the md5 of each binary of the tool."""
    print(f"python {sys.version.split()[0]} | numpy {np.__version__} | tskit {tskit.__version__} | "
          f"tsinfer {importlib.metadata.version('tsinfer')} | tsdate {importlib.metadata.version('tsdate')}",
          flush=True)
    binaries = {"relate": [RELATE_BIN, RELATE_FMT,
                           os.path.join(RELATE_SCRIPTS, "EstimatePopulationSize", "EstimatePopulationSize.sh"),
                           os.path.join(RELATE_SCRIPTS, "SampleBranchLengths", "SampleBranchLengths.sh")],
                "true_topology": [os.path.join(POLEGON_DIR, name) for name in ("polegon_master", "polegon")],
                "singer": [os.path.join(SINGER_DIR, name) for name in ("singer_master", "singer", "convert_to_tskit")]
                          + [os.path.join(POLEGON_DIR, name) for name in ("polegon_master", "polegon")]}
    for path in binaries.get(tool, []):
        if not os.path.exists(path):
            raise SystemExit(f"!! {relative(path)} does not exist")
        print(f"  {relative(path)} md5 {md5_of_file(path)}", flush=True)


def main():
    """Run the tool given as the first argument on each testing file given after the output folder, one after the
    other, and check that each data set folder holds exactly the files of OUTPUT_FILES."""
    if len(sys.argv) < 4 or sys.argv[1] not in TOOLS:
        raise SystemExit(f"usage: python run_external_tools.py {'|'.join(TOOLS)} OUT_DIR TRUE_FILE [TRUE_FILE ...]")
    tool, out_dir, paths = sys.argv[1], os.path.abspath(sys.argv[2]), sys.argv[3:]
    print(f"run_external_tools | {tool} | {len(paths)} files | output {relative(out_dir)}", flush=True)
    print_versions(tool)
    # tsdate takes about a minute to import, so the tools that use it import it here, before the first timer, and
    # the imports inside their functions then cost nothing.
    if tool in ("tsinfer", "true_topology"):
        import tsdate
        import tsinfer
    for path in paths:
        mts, row, tag = read_simulation(path)
        print(f"\n=== {tag} | {tool} | md5 {row['trees_md5']} | tool_seed {row['tool_seed']} | relate_ne "
              f"{row['relate_ne']} | diploid_ne {row['diploid_ne']} | {mts.num_samples} haplotypes, {mts.num_sites} "
              f"sites, {mts.sequence_length:.0f} bp ===", flush=True)
        folder = os.path.join(out_dir, tag)
        # A new run of the same data set and tool replaces every file of the earlier one.
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)
        steps = []
        with timed_step("task", steps):
            RUN[tool](mts, row, folder, steps)
        write_csv(os.path.join(folder, "run_times.csv"), steps)
        found, expected = sorted(os.listdir(folder)), sorted(OUTPUT_FILES[tool])
        if found != expected:
            raise SystemExit(f"!! {relative(folder)}: files {sorted(set(found) - set(expected))} not expected, "
                             f"{sorted(set(expected) - set(found))} missing")
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
