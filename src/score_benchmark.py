"""
score_benchmark.py: the benchmark metrics of one external tool on testing files
================================================================================
Scores the ARGs that src/run_external_tools.py inferred from testing files of src/simulate_data.py against the
simulated ARGs, with the metrics of src/arg_metrics.py, and writes every value of one testing file to one gzipped
table. The script needs no other script of src/ than those two.

Configurations, by tool:
  relate   relate: relate.trees, the trees that EstimatePopulationSize re-dated. The 100 draws of
           SampleBranchLengths give its calibration, and no other metric.
  tsinfer  tsinfer_tsdate: tsinfer.trees, dated by tsdate, whose posterior means are raised where needed so that each
           parent is older than its children. The gamma posterior of each node before that step
           (tsinfer_node_posteriors.npz) gives its calibration, through the rank fractions of M = 100 independent
           draws from each posterior.
  true_topology  The references on the simulated topology with its mutations placed again by parsimony.
           true_topology_tsdate: true_topology.trees, dated by tsdate and calibrated as for tsinfer.
           true_topology_polegon: the same tree sequence with POLEGON's posterior means (true_topology_polegon.npz),
           calibrated by POLEGON's 100 draws of the MRCA node of each simulated value.
  singer   singer_polegon: the M = 100 posterior samples of SINGER, each dated by POLEGON's posterior means
           (singer_polegon.npz). The point estimate of a pairwise TMRCA is its mean over the samples. The counts of
           the distribution of the times, and the metrics of one tree sequence (windows, mutation ages, and topology),
           are means over the samples. The samples with POLEGON's last draw on each give the calibration.

Outputs: OUT_DIR/{model}_S{size}_{replicate}.csv.gz for each testing file, one row per value, with the columns model,
size, replicate, configuration, quantity, pair_class, index, index_2, and value. The configuration is "simulation" for
the values of the simulated ARG and of its testing file, and the configuration of the tool for the others, so that a
quantity has one name for both. pair_class is all, within_{population}, or between_{population}_{population} for the
values of a pair class, and empty for the others. index is the time bin b, the Jeffreys bin q (1 to 20), the rank r
(0 to M), the window w (the window [10^4 (w - 1), 10^4 w), so that w starts at 1 only when s_min = 0), or the sample
of the chain, and index_2 is the inferred time bin of a 2D histogram. The quantities:
  simulation     n, P, K, s_min, s_max, sites, W, N_a, KC_rand_sum, tau (index b, 0 to 31), jeffreys_edge (index q,
                 the 19 inner edges), b_max (per pair class), and the numeric columns of the row of the testing file
                 in the table of simulate_data.py but trees, which counts the trees of the whole sequence
  both           c_q (index q), histogram_count (index b, 0 to 30), N, c_tau (index b, 0 to 30), and lambda_b (index
                 b, 0 to 29) per pair class, pi_w and D_w (index w), and trees
  configuration  tree_sequences (1, or the samples of SINGER) and M (the samples or draws of the calibration); sum_e,
                 sum_e2, RMSLE, bias, and L2 per pair class; r_bar, rho_bar, and K_prime; N_b, sum_e_b, and sum_e2_b
                 by simulated and by inferred time (N_b_by_simulated_time, ..., sum_e2_b_by_inferred_time, index b,
                 0 to 30); J; histogram_2d_count (index b and index_2); RMSLE_pi, bias_pi, W_prime_pi, RMSLE_D,
                 bias_D, and W_prime_D; N_a, RMSLE_a, bias_a, r_a, rho_a, and histogram_2d_count_a; f_r (index r),
                 KL, coverage_50, coverage_70, and coverage_90; RF, KC, KC_sum, WT, mean_arity, and polytomies;
                 wall_seconds_{step} for each step of run_times.csv; and for SINGER, incompatible_sites and
                 diversity_mse of each sample of the chain (index: the sample) from singer_trace.csv
A testing file whose table exists is skipped. A testing file that fails is reported and the next one is scored, and
the script then ends with an error that lists the failures.

Run with:
  python -u src/score_benchmark.py TOOL TOOL_DIR OUT_DIR TRUE_FILE [TRUE_FILE ...]
TOOL is relate, tsinfer, true_topology, or singer, and TOOL_DIR the folder of run_external_tools.py that holds one
folder per testing file, external_data/{tool}/testing_data/{model}.
"""

import csv
import gzip
import io
import os
import re
import sys
import time
import traceback

import numpy as np
import tskit

import arg_metrics as A
import run_external_tools as R

# The configurations of each tool.
CONFIGURATIONS = {"relate": ("relate",), "tsinfer": ("tsinfer_tsdate",),
                  "true_topology": ("true_topology_tsdate", "true_topology_polegon"), "singer": ("singer_polegon",)}
# Number of independent draws whose expected rank fractions calibrate tsdate's gamma posteriors.
GAMMA_DRAWS = 100
# The numeric columns of the table of simulate_data.py kept with the simulation, as they are written there. The column
# trees is left out, since the quantity trees holds the number of trees between s_min and s_max.
TABLE_COLUMNS = ("ancestry_seed", "mutation_seed", "tool_seed", "sequence_length", "removed_var_sites",
                 "qced_var_sites", "pi_site", "pi_branch", "t2_branch", "t2_msprime", "relate_ne", "diploid_ne")
COLUMNS = ("model", "size", "replicate", "configuration", "quantity", "pair_class", "index", "index_2", "value")
# A token of Relate's newick trees: a bracket, a comma, a semicolon, a branch length, or a leaf with its branch length.
NEWICK_TOKEN = re.compile(r"[(),;]|:[^(),;]+|[^(),;:]+(?::[^(),;]+)?")


def format_value(value):
    """value as text: an integer as it is, and a float with 12 significant digits, nan, or inf."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.12g}"


def add(rows, configuration, quantity, value, pair_class="", index="", index_2=""):
    """Append the row (configuration, quantity, pair_class, index, index_2, value) to rows."""
    rows.append((configuration, quantity, pair_class, index, index_2, format_value(value)))


def add_array(rows, configuration, quantity, values, first_index, pair_class=""):
    """Append one row per value of an array, the first with index first_index."""
    for offset, value in enumerate(values):
        add(rows, configuration, quantity, value, pair_class, first_index + offset)


def add_2d(rows, configuration, quantity, counts):
    """Append one row per nonzero count of a 2D histogram, with the simulated bin as index and the inferred bin as
    index_2."""
    for b, b_hat in zip(*np.nonzero(counts)):
        add(rows, configuration, quantity, counts[b, b_hat], "", int(b), int(b_hat))


def simulation_values(simulation):
    """Everything about the simulated ARG that the metrics compare against, computed once per testing file: the
    positions x, t [K, P], the pair classes with N, c_tau, lambda_b, and b_max of each, the windows with pi_w and D_w,
    the mutation ages a, the Jeffreys edges and c_q, the histogram, the triples with their labels, KC_rand_sum, and
    the number of trees between s_min and s_max."""
    x = A.evaluation_positions(simulation)
    summary = A.tree_summaries(simulation, x)
    t = simulation.nodes_time[summary["mrca"]]
    s_min, s_max = simulation.tables.sites.position[[0, -1]]
    classes = A.pair_classes(simulation)
    N = {name: int(t[:, mask].size) for name, mask in classes.items()}
    c_tau = {name: A.older_counts(t[:, mask]) for name, mask in classes.items()}
    starts = A.window_starts(simulation)
    pi_w, D_w = A.window_statistics(simulation, starts, R.MUT_RATE)
    edges = A.jeffreys_edges(t)
    pairs = A.triple_pairs(simulation.num_samples)
    # The label of every triple at every position, computed once for every configuration.
    labels = np.concatenate([A.triple_labels(summary["clade_size"][start:start + A.CHUNK], pairs)
                             for start in range(0, len(x), A.CHUNK)])
    a = A.mutation_ages(simulation)
    return {"ts": simulation, "n": simulation.num_samples, "P": t.shape[1], "K": len(x), "x": x, "t": t,
            "s_min": float(s_min), "s_max": float(s_max), "sites": simulation.num_sites, "classes": classes, "N": N,
            "c_tau": c_tau, "lambda_b": {name: A.coalescence_rates(A.survival(c_tau[name], N[name])) for name in N},
            "b_max": {name: A.last_rate_bin(c_tau[name], N[name]) for name in N}, "starts": starts, "W": len(starts),
            "pi_w": pi_w, "D_w": D_w, "a": a, "N_a": len(a), "jeffreys_edges": edges, "c_q": A.bin_counts(t, edges),
            "histogram_count": A.histogram_counts(t), "pairs": pairs, "labels": labels,
            "KC_rand_sum": A.random_tree_kc(simulation, x), "trees": A.number_of_trees(simulation, s_min, s_max)}


def window_index(simulated):
    """The window index w of each 10 kb window, the window [10^4 (w - 1), 10^4 w)."""
    return (simulated["starts"] // A.WINDOW_SIZE).astype(np.int64) + 1


def simulation_rows(simulated, row):
    """The rows of the simulated ARG and of its testing file, configuration "simulation"."""
    rows, name = [], "simulation"
    for quantity in ("n", "P", "K", "s_min", "s_max", "sites", "W", "N_a", "KC_rand_sum", "trees"):
        add(rows, name, quantity, simulated[quantity])
    # The columns of the table as they are written there.
    rows += [(name, column, "", "", "", row[column]) for column in TABLE_COLUMNS]
    add_array(rows, name, "tau", A.TAU, 0)
    add_array(rows, name, "jeffreys_edge", simulated["jeffreys_edges"], 1)
    add_array(rows, name, "c_q", simulated["c_q"], 1)
    add_array(rows, name, "histogram_count", simulated["histogram_count"], 0)
    for pair_class in simulated["classes"]:
        add(rows, name, "N", simulated["N"][pair_class], pair_class)
        add_array(rows, name, "c_tau", simulated["c_tau"][pair_class], 0, pair_class)
        add_array(rows, name, "lambda_b", simulated["lambda_b"][pair_class], 0, pair_class)
        add(rows, name, "b_max", simulated["b_max"][pair_class], pair_class)
    for w, pi, D in zip(window_index(simulated), simulated["pi_w"], simulated["D_w"]):
        add(rows, name, "pi_w", pi, "", int(w))
        add(rows, name, "D_w", D, "", int(w))
    return rows


def new_accumulator(simulated):
    """Running sums of the values of the tree sequences of one configuration."""
    return {"tree_sequences": 0, "t_hat_sum": np.zeros_like(simulated["t"]), "c_q": np.zeros(A.JEFFREYS_BINS),
            "histogram_count": np.zeros(A.TIME_BINS), "c_tau": {name: np.zeros(A.TIME_BINS) for name in simulated["N"]},
            "pi_w": np.zeros(simulated["W"]), "D_w": np.zeros(simulated["W"]), "a_hat_sum": {}, "a_hat_count": {},
            "RF": [], "KC_sum": [], "WT": [], "mean_arity": [], "polytomies": [], "trees": []}


def add_tree_sequence(accumulator, ts, dated, simulated):
    """Add the values of one inferred tree sequence to the accumulator of its configuration, and return its MRCA nodes
    [K, P]. ts gives the topology and dated the node times: the same tree sequence, or a SINGER sample with POLEGON's
    ages, whose nodes are those of ts."""
    summary = A.tree_summaries(ts, simulated["x"])
    t_hat = dated.nodes_time[summary["mrca"]]
    accumulator["tree_sequences"] += 1
    accumulator["t_hat_sum"] += t_hat
    # The distribution of the sampled times, summed over the tree sequences.
    accumulator["c_q"] += A.bin_counts(t_hat, simulated["jeffreys_edges"])
    accumulator["histogram_count"] += A.histogram_counts(t_hat)
    for name, mask in simulated["classes"].items():
        accumulator["c_tau"][name] += A.older_counts(t_hat[:, mask])
    pi_w, D_w = A.window_statistics(dated, simulated["starts"], R.MUT_RATE)
    accumulator["pi_w"] += pi_w
    accumulator["D_w"] += D_w
    # The age of each site, summed over the tree sequences in which it carries exactly one mutation.
    for key, age in A.mutation_ages(dated).items():
        accumulator["a_hat_sum"][key] = accumulator["a_hat_sum"].get(key, 0.0) + age
        accumulator["a_hat_count"][key] = accumulator["a_hat_count"].get(key, 0) + 1
    distances = A.tree_distances(ts, simulated["ts"], simulated["x"])
    accumulator["RF"].append(distances["RF"])
    accumulator["KC_sum"].append(distances["KC_sum"])
    accumulator["WT"].append(A.wrong_triplets(simulated["labels"], summary["clade_size"], simulated["pairs"]))
    accumulator["mean_arity"].append(float(summary["arity"].mean()))
    accumulator["polytomies"].append(float(summary["polytomies"].mean()))
    accumulator["trees"].append(A.number_of_trees(ts, simulated["s_min"], simulated["s_max"]))
    return summary["mrca"]


def configuration_rows(configuration, accumulator, simulated):
    """The rows of every metric of one configuration but its calibration, from the accumulator of its tree sequences.
    The point estimate t_hat is the mean of the pairwise TMRCA over the tree sequences, and the counts of the
    distribution of the times and the metrics of one tree sequence are means over them."""
    rows, count = [], accumulator["tree_sequences"]
    t, t_hat = simulated["t"], accumulator["t_hat_sum"] / count
    e = A.log_errors(t_hat, t)
    add(rows, configuration, "tree_sequences", count)
    for pair_class, mask in simulated["classes"].items():
        summary = A.error_summary(e[:, mask])
        for quantity in ("N", "sum_e", "sum_e2", "RMSLE", "bias"):
            add(rows, configuration, quantity, summary[quantity], pair_class)
        c_tau = accumulator["c_tau"][pair_class] / count
        lambda_hat_b = A.coalescence_rates(A.survival(c_tau, simulated["N"][pair_class]))
        add_array(rows, configuration, "c_tau", c_tau, 0, pair_class)
        add_array(rows, configuration, "lambda_b", lambda_hat_b, 0, pair_class)
        add(rows, configuration, "L2", A.rate_l2(lambda_hat_b, simulated["lambda_b"][pair_class],
                                                simulated["b_max"][pair_class]), pair_class)
    for quantity, value in A.position_correlations(t_hat, t).items():
        add(rows, configuration, quantity, value)
    for binned_by, sums in A.errors_by_time(e, t, t_hat).items():
        for quantity, values in zip(("N_b", "sum_e_b", "sum_e2_b"), sums):
            add_array(rows, configuration, f"{quantity}_by_{binned_by}_time", values, 0)
    c_hat_q = accumulator["c_q"] / count
    add(rows, configuration, "J", A.jeffreys_divergence(simulated["c_q"], c_hat_q))
    add_array(rows, configuration, "c_q", c_hat_q, 1)
    add_array(rows, configuration, "histogram_count", accumulator["histogram_count"] / count, 0)
    add_2d(rows, configuration, "histogram_2d_count", A.histogram_2d_counts(t, t_hat))
    pi_hat_w, D_hat_w = accumulator["pi_w"] / count, accumulator["D_w"] / count
    for w, pi, D in zip(window_index(simulated), pi_hat_w, D_hat_w):
        add(rows, configuration, "pi_w", pi, "", int(w))
        add(rows, configuration, "D_w", D, "", int(w))
    for statistic, estimate, simulated_w in (("pi", pi_hat_w, simulated["pi_w"]),
                                             ("D", D_hat_w, simulated["D_w"])):
        for quantity, value in A.window_errors(estimate, simulated_w).items():
            add(rows, configuration, f"{quantity}_{statistic}", value)
    a_hat = {key: total / accumulator["a_hat_count"][key] for key, total in accumulator["a_hat_sum"].items()}
    age_scores, age_counts = A.mutation_age_scores(simulated["a"], a_hat)
    add(rows, configuration, "N_a", age_scores["N_a"])
    for quantity in ("RMSLE", "bias", "r", "rho"):
        add(rows, configuration, f"{quantity}_a", age_scores[quantity])
    add_2d(rows, configuration, "histogram_2d_count_a", age_counts)
    add(rows, configuration, "RF", float(np.mean(accumulator["RF"])))
    add(rows, configuration, "KC", float(np.mean(accumulator["KC_sum"])) / simulated["KC_rand_sum"])
    for quantity in ("KC_sum", "WT", "mean_arity", "polytomies", "trees"):
        add(rows, configuration, quantity, float(np.mean(accumulator[quantity])))
    return rows


def calibration_rows(configuration, f_r, covered, N, M):
    """The rows of the calibration of one configuration: M, the rank fractions f_r, KL, and the coverage of the
    levels gamma, from covered, {gamma: the number of simulated values inside the interval}, and N."""
    rows = []
    add(rows, configuration, "M", M)
    add_array(rows, configuration, "f_r", f_r, 0)
    add(rows, configuration, "KL", A.kl_from_uniform(f_r))
    for gamma in A.GAMMA:
        add(rows, configuration, f"coverage_{round(100 * gamma)}", covered[gamma] / N)
    return rows


def newick_tmrca_matrix(newick, n):
    """[n, n] pairwise TMRCA of one newick tree of Relate, with leaves named 0 to n - 1 and branch lengths in
    generations. A node's age is its first child's age plus that child's branch length, and every pair of leaves
    below two different children of a node gets that node's age."""
    tmrca = np.zeros((n, n))
    # Children of each open node: [leaves, age, branch length].
    stack = [[]]
    for token in NEWICK_TOKEN.findall(newick):
        if token == "(":
            stack.append([])
        elif token == ")":
            kids = stack.pop()
            age = kids[0][1] + kids[0][2]
            for a in range(len(kids)):
                for b in range(a + 1, len(kids)):
                    leaves_a, leaves_b = np.array(kids[a][0]), np.array(kids[b][0])
                    tmrca[np.ix_(leaves_a, leaves_b)] = age
                    tmrca[np.ix_(leaves_b, leaves_a)] = age
            stack[-1].append([[leaf for kid in kids for leaf in kid[0]], age, 0.0])
        elif token.startswith(":"):
            # Branch length of the node just closed.
            stack[-1][-1][2] = float(token[1:])
        elif token not in (",", ";"):
            name, _, branch = token.partition(":")
            stack[-1].append([[int(name)], 0.0, float(branch or 0)])
    return tmrca


def relate_draw_times(prefix, simulated):
    """[M, K, P] pairwise TMRCA (float32) of the M draws of SampleBranchLengths at the evaluation positions, from
    {prefix}.newick.gz, one line per tree and draw (chromosome, start, end, draw, and the newick tree), and the REGION
    line of {prefix}.sites. Positions outside the region are moved to its ends, and every position of every draw must
    get a tree."""
    with open(prefix + ".sites") as fh:
        region = next((line.split() for line in fh if line.startswith("REGION")), None)
    if region is None:
        raise ValueError(f"{os.path.basename(prefix)}.sites has no REGION line")
    x = np.clip(simulated["x"], float(region[2]), float(region[3]) - 1)
    first, second = np.triu_indices(simulated["n"], 1)
    times = np.full((R.RELATE_DRAWS, len(x), len(first)), np.nan, dtype=np.float32)
    with gzip.open(prefix + ".newick.gz", "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            _, start, end, draw, newick = line.rstrip("\n").split("\t", 4)
            # Evaluation positions inside this tree.
            lo, hi = np.searchsorted(x, [float(start), float(end)], side="left")
            if hi > lo:
                times[int(draw), lo:hi] = newick_tmrca_matrix(newick, simulated["n"])[first, second]
    missing = sum(int(np.isnan(draw_times).any(axis=1).sum()) for draw_times in times)
    if missing:
        raise ValueError(f"the draws of SampleBranchLengths have no tree at {missing} positions")
    return times


def score_relate(folder, simulated):
    """The rows of Relate: every metric of relate.trees, and the calibration by the draws of SampleBranchLengths."""
    configuration = CONFIGURATIONS["relate"][0]
    ts = tskit.load(os.path.join(folder, "relate.trees"))
    accumulator = new_accumulator(simulated)
    add_tree_sequence(accumulator, ts, ts, simulated)
    rows = configuration_rows(configuration, accumulator, simulated)
    draws = relate_draw_times(os.path.join(folder, "relate_sample_branch_lengths"), simulated)
    chunks = ((draws[:, start:start + A.CHUNK], simulated["t"][start:start + A.CHUNK])
              for start in range(0, simulated["K"], A.CHUNK))
    rank_counts, covered, N = A.sample_ranks_and_coverage(chunks)
    return rows + calibration_rows(configuration, rank_counts / N, covered, N, draws.shape[0])


def score_tsdate(folder, simulated, name, configuration):
    """The rows of a tree sequence dated by tsdate, {name}.trees: every metric, and the calibration by tsdate's gamma
    posteriors ({name}_node_posteriors.npz), with the expected rank fractions of GAMMA_DRAWS independent draws. The
    tree sequence and its MRCA nodes are also returned."""
    ts = tskit.load(os.path.join(folder, f"{name}.trees"))
    accumulator = new_accumulator(simulated)
    mrca = add_tree_sequence(accumulator, ts, ts, simulated)
    rows = configuration_rows(configuration, accumulator, simulated)
    posteriors = np.load(os.path.join(folder, f"{name}_node_posteriors.npz"))
    shape, rate = A.gamma_posteriors(posteriors["mean"], posteriors["variance"])
    U, covered = A.gamma_calibration(shape, rate, mrca, simulated["t"])
    rows += calibration_rows(configuration, A.binomial_rank_fractions(U, GAMMA_DRAWS), covered, U.size, GAMMA_DRAWS)
    return rows, ts, mrca


def score_tsinfer(folder, simulated):
    """The rows of tsinfer + tsdate, those of score_tsdate for tsinfer.trees."""
    return score_tsdate(folder, simulated, "tsinfer", CONFIGURATIONS["tsinfer"][0])[0]


def score_true_topology(folder, simulated):
    """The rows of the two references on the simulated topology with its mutations placed again by parsimony: tsdate,
    those of score_tsdate for true_topology.trees, and POLEGON, the same tree sequence with POLEGON's posterior means
    (true_topology_polegon.npz), calibrated by POLEGON's draws of the MRCA node of each simulated value. tsdate keeps
    the nodes of its input, so POLEGON's ages, which follow the node order of that input, fit true_topology.trees."""
    tsdate_configuration, polegon_configuration = CONFIGURATIONS["true_topology"]
    rows, ts, mrca = score_tsdate(folder, simulated, "true_topology", tsdate_configuration)
    polegon = np.load(os.path.join(folder, "true_topology_polegon.npz"))
    accumulator = new_accumulator(simulated)
    add_tree_sequence(accumulator, ts, polegon_dated(ts, polegon["mean"].astype(np.float64)), simulated)
    rows += configuration_rows(polegon_configuration, accumulator, simulated)
    draws = polegon["draws"].astype(np.float64)
    rank_counts, covered, N = A.draw_ranks_and_coverage(draws, mrca, simulated["t"])
    return rows + calibration_rows(polegon_configuration, rank_counts / N, covered, N, draws.shape[1])


def polegon_dated(ts, ages):
    """ts with POLEGON's ages as its node times, in generations, with the times of the mutations left unknown. tskit
    raises if a parent is not older than its child. Sorting the tables keeps the node IDs."""
    tables = ts.dump_tables()
    tables.nodes.time = ages
    tables.time_units = "generations"
    tables.mutations.time = np.full(tables.mutations.num_rows, tskit.UNKNOWN_TIME)
    tables.sort()
    return tables.tree_sequence()


def trace_rows(folder, configuration):
    """The rows of SINGER's trace: incompatible_sites and diversity_mse of each sample of the chain."""
    rows = []
    with open(os.path.join(folder, "singer_trace.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            add(rows, configuration, "incompatible_sites", int(row["incompatible_sites"]), "", int(row["sample"]))
            add(rows, configuration, "diversity_mse", float(row["diversity_mse"]), "", int(row["sample"]))
    return rows


def score_singer(folder, simulated):
    """The rows of SINGER + POLEGON: every metric of the M posterior samples, each dated by POLEGON's posterior means,
    the calibration by the samples with POLEGON's last draw on each, and the trace of the chain. The MRCA nodes of
    every sample are kept, so that the calibration can take the samples of a chunk of positions together."""
    configuration = CONFIGURATIONS["singer"][0]
    polegon = np.load(os.path.join(folder, "singer_polegon.npz"))
    M = len(R.SINGER_KEPT)
    accumulator = new_accumulator(simulated)
    mrca = np.empty((M,) + simulated["t"].shape, dtype=np.int32)
    last_draws = []
    for m, k in enumerate(R.SINGER_KEPT):
        ts = tskit.load(os.path.join(folder, f"singer_{k}.trees"))
        dated = polegon_dated(ts, polegon[f"mean_{k}"].astype(np.float64))
        mrca[m] = add_tree_sequence(accumulator, ts, dated, simulated)
        last_draws.append(polegon[f"last_draw_{k}"].astype(np.float64))
    rows = configuration_rows(configuration, accumulator, simulated)
    # The samples of each simulated value: its pairwise TMRCA in the M samples, with POLEGON's last draws as times.
    chunks = ((np.stack([draw[mrca[m, start:start + A.CHUNK]] for m, draw in enumerate(last_draws)]),
               simulated["t"][start:start + A.CHUNK]) for start in range(0, simulated["K"], A.CHUNK))
    rank_counts, covered, N = A.sample_ranks_and_coverage(chunks)
    return rows + calibration_rows(configuration, rank_counts / N, covered, N, M) + trace_rows(folder, configuration)


SCORE = {"relate": score_relate, "tsinfer": score_tsinfer, "true_topology": score_true_topology,
         "singer": score_singer}


def run_time_rows(folder, configuration):
    """The rows of the wall time in seconds of each step of the tool, wall_seconds_{step}, from run_times.csv."""
    rows = []
    with open(os.path.join(folder, "run_times.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            add(rows, configuration, f"wall_seconds_{row['step']}", float(row["wall_seconds"]))
    return rows


def write_table(path, row, rows):
    """Write the rows of one testing file, after its model, size, and replicate, as a gzipped table. The table goes
    through a temporary file, so that an interrupted job leaves no table, and its gzip header holds no time and no
    file name, so that the same rows give the same bytes."""
    temporary = path + ".tmp"
    with (open(temporary, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
          io.TextIOWrapper(compressed, encoding="utf-8", newline="") as fh):
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(COLUMNS)
        prefix = (row["model"], row["size"], row["replicate"])
        writer.writerows(prefix + values for values in rows)
    os.replace(temporary, path)


def value_of(rows, configuration, quantity, pair_class=""):
    """The value of one row of rows, as text, for the line that the script prints per testing file."""
    return next(values[5] for values in rows if values[:3] == (configuration, quantity, pair_class))


def summary_text(rows, configuration):
    """The metrics of one configuration that the script prints per testing file."""
    return (f"RMSLE {value_of(rows, configuration, 'RMSLE', 'all')}, "
            f"bias {value_of(rows, configuration, 'bias', 'all')}, rho_bar {value_of(rows, configuration, 'rho_bar')}, "
            f"J {value_of(rows, configuration, 'J')}, L2 {value_of(rows, configuration, 'L2', 'all')}, "
            f"KL {value_of(rows, configuration, 'KL')}, WT {value_of(rows, configuration, 'WT')}")


def main():
    """Score one tool on the testing files of the command line."""
    if len(sys.argv) < 5 or sys.argv[1] not in CONFIGURATIONS:
        raise SystemExit("!! usage: python -u src/score_benchmark.py TOOL TOOL_DIR OUT_DIR TRUE_FILE [TRUE_FILE ...], "
                         f"with TOOL one of {', '.join(CONFIGURATIONS)}")
    tool, tool_dir, out_dir, true_files = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    configurations = CONFIGURATIONS[tool]
    print(f"score_benchmark | {tool} | {len(true_files)} testing files | numpy {np.__version__} | tskit "
          f"{tskit.__version__}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    failed = []
    for true_file in true_files:
        tag = os.path.basename(true_file)[:-len(".trees")]
        path = os.path.join(out_dir, f"{tag}.csv.gz")
        if os.path.exists(path):
            print(f"=== {tag} | table exists, skipped", flush=True)
            continue
        start = time.perf_counter()
        # One testing file that fails must not stop the others, so any error is reported and the next file scored.
        try:
            simulation, row, tag = R.read_simulation(true_file)
            simulated = simulation_values(simulation)
            folder = os.path.join(tool_dir, tag)
            rows = simulation_rows(simulated, row) + SCORE[tool](folder, simulated)
            # The run times of the tool, with each of its configurations.
            for configuration in configurations:
                rows += run_time_rows(folder, configuration)
            write_table(path, row, rows)
        except (Exception, SystemExit) as error:
            failed.append(tag)
            print(f"!! {tag} | {type(error).__name__}: {R.relative(str(error))}", flush=True)
            print(R.relative(traceback.format_exc()), flush=True)
            continue
        print(f"=== {tag} | {len(rows)} rows, {os.path.getsize(path) / 1e3:.0f} kB, "
              f"{time.perf_counter() - start:.0f} s | {summary_text(rows, configurations[0])}", flush=True)
        for configuration in configurations[1:]:
            print(f"    {configuration} | {summary_text(rows, configuration)}", flush=True)
    if failed:
        raise SystemExit(f"!! {len(failed)} of {len(true_files)} testing files failed: {' '.join(failed)}")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
