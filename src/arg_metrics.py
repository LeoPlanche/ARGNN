"""
arg_metrics.py: the metrics of the benchmark of inferred ARGs
==============================================================
Library for score_benchmark.py, with no file reading or writing: each function takes arrays or tree sequences and
returns numbers or arrays. The sections follow the metrics: positions and trees, pairwise TMRCA, coalescence rates,
statistics in genomic windows, mutation ages, calibration, and topology.

Notation: n haplotypes, the sample nodes 0 to n - 1, and P = n(n - 1)/2 pairs (i, j) with i < j, in the order of
numpy.triu_indices(n, 1). K evaluation positions x_k lie between s_min and s_max, the positions of the first and the
last site of the simulation. t and t_hat are [K, P] arrays of the simulated and the inferred pairwise TMRCA in
generations, and e = ln(1 + t_hat) - ln(1 + t). M is the number of posterior samples or draws. TAU is the time grid
tau_0 = 0, tau_b = 100 x 2000^((b - 1)/29) for b = 1 to 30, and tau_31 = infinity, and bin b is [tau_b, tau_{b+1})
for b = 0 to 30. A name keeps the case of its symbol, with a hat, a bar, a prime, or a subscript after an underscore
(t_hat, r_bar, K_prime, N_b).

Package functions do the work wherever one exists: scipy.stats for the correlations, the divergences, and the gamma
posteriors, scipy.special for the binomial coefficients, numpy for the counts and the quantiles, and tskit for the
window statistics, the random splits of polytomies, the random binary trees, and the Robinson-Foulds and
Kendall-Colijn distances.
"""

import warnings
from itertools import combinations

import numpy as np
import tskit
from scipy import special, stats

# Number of evaluation positions.
K = 1000
# Length of the genomic windows in base pairs.
WINDOW_SIZE = 10_000
# The time grid: tau_0 = 0, 30 times evenly spaced on a log scale from 100 to 200,000 generations, and tau_31 = inf.
TAU = np.concatenate([[0.0], 100 * 2000.0 ** (np.arange(30) / 29), [np.inf]])
# Bins of the time grid, b = 0 to 30. The coalescence rates use bins 0 to 29, whose ends are finite.
TIME_BINS = len(TAU) - 1
RATE_BINS = TIME_BINS - 1
# Number of bins of the Jeffreys divergence, Q_1 to Q_20.
JEFFREYS_BINS = 20
# The L2 of the coalescence rates stops at the last bin whose end has at least this fraction of the simulated values
# older.
L2_MIN_SURVIVAL = 0.01
# Levels gamma of the coverage.
GAMMA = (0.5, 0.7, 0.9)
# Number of positions per chunk in the functions that build [positions, pairs, ...] arrays.
CHUNK = 50


def evaluation_positions(simulated_ts):
    """The K evaluation positions x_k = s_min + (k - 1/2)(s_max - s_min)/K for k = 1 to K, the midpoints of K equal
    bins between s_min and s_max, the positions of the first and the last site of the simulation, simulated_ts."""
    s_min, s_max = simulated_ts.tables.sites.position[[0, -1]]
    k = np.arange(1, K + 1)
    return s_min + (k - 0.5) * (s_max - s_min) / K


def tree_summaries(ts, x):
    """For the tree of ts at each position of x: mrca and clade_size, [K, P] arrays (int32) of the MRCA node of each
    pair and of the number of samples below it, arity [K], the mean number of children of the nodes with at least
    two children, and polytomies [K], the number of nodes with more than two children (wohns2022unified). One
    postorder pass per distinct tree: an internal node u is the MRCA of every pair of samples below two different
    children of u, so each pair gets its MRCA once. No tskit function gives the MRCA node of every pair at once.
    Positions in the same tree share its results. A pair without an MRCA, as in a tree with two roots, raises a
    ValueError."""
    n = ts.num_samples
    if not np.array_equal(ts.samples(), np.arange(n)):
        raise ValueError("the sample nodes must be 0 to n - 1")
    first, second = np.triu_indices(n, 1)
    # Pair index of every ordered pair of samples, so that a block of pairs can be addressed at once.
    pair_index = np.zeros((n, n), dtype=np.int64)
    pair_index[first, second] = np.arange(len(first))
    pair_index[second, first] = np.arange(len(first))
    mrca = np.empty((len(x), len(first)), dtype=np.int32)
    clade_size = np.empty((len(x), len(first)), dtype=np.int32)
    arity, polytomies = np.empty(len(x)), np.empty(len(x), dtype=np.int64)
    tree = tskit.Tree(ts)
    k = 0
    while k < len(x):
        tree.seek(x[k])
        # Positions that fall in this tree.
        end = int(np.searchsorted(x, tree.interval.right, side="left"))
        row_node = np.full(len(first), -1, dtype=np.int32)
        row_size = np.zeros(len(first), dtype=np.int32)
        below, arities = {}, []
        for u in tree.nodes(order="postorder"):
            children = tree.children(u)
            if not children:
                below[u] = np.array([u] if tree.is_sample(u) else [], dtype=np.int64)
                continue
            kids = [below.pop(c) for c in children]
            count = sum(len(kid) for kid in kids)
            # Every pair of samples below two different children has u as its MRCA.
            for a, b in combinations(range(len(kids)), 2):
                if len(kids[a]) and len(kids[b]):
                    index = pair_index[np.ix_(kids[a], kids[b])].ravel()
                    row_node[index] = u
                    row_size[index] = count
            below[u] = np.concatenate(kids)
            if len(children) >= 2:
                arities.append(len(children))
        if (row_node < 0).any():
            raise ValueError(f"{(row_node < 0).sum()} pairs have no MRCA at position {x[k]} ({tree.num_roots} roots)")
        mrca[k:end], clade_size[k:end] = row_node, row_size
        arity[k:end] = np.mean(arities)
        polytomies[k:end] = sum(a > 2 for a in arities)
        k = end
    return {"mrca": mrca, "clade_size": clade_size, "arity": arity, "polytomies": polytomies}


def pair_classes(simulated_ts):
    """The pair classes, {name: boolean mask over the P pairs}: all pairs ("all"), and the pairs within one sampled
    population ("within_{name}") or between two ("between_{name}_{name}") when more than one population has samples,
    with the population names of the metadata of simulated_ts."""
    n = simulated_ts.num_samples
    first, second = np.triu_indices(n, 1)
    population = simulated_ts.nodes_population[:n]
    names = {p.id: p.metadata["name"] if isinstance(p.metadata, dict) and "name" in p.metadata else str(p.id)
             for p in simulated_ts.populations()}
    sampled = sorted(set(population.tolist()))
    classes = {"all": np.ones(len(first), dtype=bool)}
    if len(sampled) > 1:
        for a in sampled:
            classes[f"within_{names[a]}"] = (population[first] == a) & (population[second] == a)
        for a, b in combinations(sampled, 2):
            classes[f"between_{names[a]}_{names[b]}"] = (
                ((population[first] == a) & (population[second] == b))
                | ((population[first] == b) & (population[second] == a)))
    return classes


def number_of_trees(ts, s_min, s_max):
    """The number of trees of ts that overlap [s_min, s_max]."""
    breakpoints = ts.breakpoints(as_array=True)
    return int(np.searchsorted(breakpoints, s_max, side="right") - np.searchsorted(breakpoints, s_min, side="right")
               + 1)


def log_errors(t_hat, t):
    """e = ln(1 + t_hat) - ln(1 + t), elementwise."""
    return np.log1p(t_hat) - np.log1p(t)


def error_summary(e):
    """N, the number of errors e on the log scale, sum_e and sum_e2, the sums of e and e^2, RMSLE = sqrt(sum_e2 / N),
    and bias = sum_e / N. The count and the sums let the metrics be pooled over replicates. RMSLE and bias are NaN
    when N = 0."""
    N = int(e.size)
    if N == 0:
        return {"N": 0, "sum_e": 0.0, "sum_e2": 0.0, "RMSLE": np.nan, "bias": np.nan}
    sum_e, sum_e2 = float(e.sum()), float(np.square(e).sum())
    return {"N": N, "sum_e": sum_e, "sum_e2": sum_e2, "RMSLE": float(np.sqrt(sum_e2 / N)), "bias": sum_e / N}


def position_correlations(t_hat, t):
    """r_bar and rho_bar, the means over positions of the Pearson correlation r_k and the Spearman correlation rho_k
    over the P pairs of y_hat = ln(1 + t_hat) and y = ln(1 + t), and K_prime, the number of positions where neither
    set of values is constant (yCBrandt2022evaluation). rho_k is r_k of the ranks of the values at position k, with
    tied values given the mean of their ranks (scipy.stats.rankdata). r_k and rho_k are NaN at a position with
    constant values, and numpy.nanmean leaves them out. r_bar and rho_bar are NaN when K_prime = 0."""
    y_hat, y = np.log1p(t_hat), np.log1p(t)
    with warnings.catch_warnings():
        # The function pearsonr warns once per call when a position has constant values, which then gives NaN.
        warnings.simplefilter("ignore", stats.ConstantInputWarning)
        r_k = stats.pearsonr(y_hat, y, axis=1).statistic
        rho_k = stats.pearsonr(stats.rankdata(y_hat, axis=1), stats.rankdata(y, axis=1), axis=1).statistic
    K_prime = int(np.isfinite(r_k).sum())
    if K_prime == 0:
        return {"r_bar": np.nan, "rho_bar": np.nan, "K_prime": 0}
    return {"r_bar": float(np.nanmean(r_k)), "rho_bar": float(np.nanmean(rho_k)), "K_prime": K_prime}


def time_bins(values):
    """The bin b of TAU of each value, tau_b <= value < tau_{b+1}, flattened. Times are never negative, so every
    value falls in one of the 31 bins."""
    return np.searchsorted(TAU, np.ravel(values), side="right") - 1


def errors_by_time(e, t, t_hat):
    """N_b, sum_e_b, and sum_e2_b, the number of errors e and their sums of e and e^2 in each bin b of TAU, with the
    values binned by the simulated time t ("simulated") and by the inferred time t_hat ("inferred"). Returns
    {binned_by: (N_b, sum_e_b, sum_e2_b)}, each an array over the 31 bins. RMSLE_b = sqrt(sum_e2_b / N_b) and
    bias_b = sum_e_b / N_b follow from the sums, also when they are pooled over replicates."""
    errors, squares = np.ravel(e), np.square(np.ravel(e))
    out = {}
    for binned_by, times in (("simulated", t), ("inferred", t_hat)):
        b = time_bins(times)
        out[binned_by] = (np.bincount(b, minlength=TIME_BINS), np.bincount(b, weights=errors, minlength=TIME_BINS),
                          np.bincount(b, weights=squares, minlength=TIME_BINS))
    return out


def jeffreys_edges(t):
    """The 19 inner edges of the bins Q_1 to Q_20 of the Jeffreys divergence: the quantiles 0.05, 0.10, ..., 0.95 of
    the simulated values t (deng2025general). The outer edges are 0 and infinity."""
    return np.quantile(t, np.arange(1, JEFFREYS_BINS) / JEFFREYS_BINS)


def bin_counts(values, edges):
    """The number of values in each bin of the inner edges, len(edges) + 1 counts, bin q holding the values in
    [edges[q - 1], edges[q])."""
    return np.bincount(np.searchsorted(edges, np.ravel(values), side="right"), minlength=len(edges) + 1)


def jeffreys_divergence(c_q, c_hat_q):
    """J = sum_q (p_q - p_hat_q) ln(p_q / p_hat_q), with p_q = (c_q + 1/2)/(N + 10) and p_hat_q = (c_hat_q + 1/2)/
    (N + 10) the fractions of the simulated and of the inferred values in the 20 bins, each bin with half a count
    added. For samples, c_hat_q is the mean over samples. J is the sum of the Kullback-Leibler divergences in both
    directions, each scipy.stats.entropy."""
    p_q = (c_q + 0.5) / (np.sum(c_q) + 0.5 * len(c_q))
    p_hat_q = (c_hat_q + 0.5) / (np.sum(c_hat_q) + 0.5 * len(c_hat_q))
    return float(stats.entropy(p_q, p_hat_q) + stats.entropy(p_hat_q, p_q))


def histogram_counts(values):
    """The number of values in each of the 31 bins of TAU."""
    return np.bincount(time_bins(values), minlength=TIME_BINS)


def histogram_2d_counts(t, t_hat):
    """The 2D histogram of the pairs (t, t_hat) on the bins of TAU, [31, 31] counts, the simulated bin first."""
    pairs = time_bins(t) * TIME_BINS + time_bins(t_hat)
    return np.bincount(pairs, minlength=TIME_BINS * TIME_BINS).reshape(TIME_BINS, TIME_BINS)


def older_counts(values):
    """c_tau, the number c(tau_b) of values older than tau_b, for the 31 finite times tau_0 to tau_30 of TAU. A value
    equal to tau_b is not older than it."""
    ordered = np.sort(np.ravel(values))
    return len(ordered) - np.searchsorted(ordered, TAU[:-1], side="right")


def survival(c_tau, N):
    """S_half(tau_b) = (c(tau_b) + 1/2)/(N + 1), the survival function of N values with half a count for each of
    the two outcomes, older than tau_b or not. It is the mean of the posterior Beta(c + 1/2, N - c + 1/2) of the
    fraction under the Jeffreys prior, never 0, and differs from S = c/N by at most 1/(2(N + 1))."""
    return (np.asarray(c_tau, dtype=float) + 0.5) / (N + 1)


def coalescence_rates(S_half):
    """lambda_b = -(ln S_half(tau_{b+1}) - ln S_half(tau_b))/(tau_{b+1} - tau_b) for the 30 bins b = 0 to 29, from
    the survival function at tau_0 to tau_30 (deng2025general, Eq. 7). The effective population size of bin b is
    N_e_b = 1/(2 lambda_b). Where the values run out inside bin b, with c values older than tau_b and none older than
    tau_{b+1}, lambda_b = ln(2c + 1)/(tau_{b+1} - tau_b), and every later rate is 0."""
    log_S_half = np.log(S_half)
    # Adding 0.0 turns the -0.0 of a bin without coalescence into 0.0, so that 1/(2 lambda_b) is +inf there.
    return -(log_S_half[1:] - log_S_half[:-1]) / np.diff(TAU[:-1]) + 0.0


def last_rate_bin(c_tau, N):
    """b_max, the last bin b = 0 to 29 whose end has at least L2_MIN_SURVIVAL of the simulated values older:
    S(tau_{b_max+1}) >= 0.01 > S(tau_{b_max+2}), with S = c/N the simulated survival function without half counts.
    S does not increase, so the bins that pass are bins 0 to b_max. b_max is -1 when S(tau_1) < 0.01."""
    S = np.asarray(c_tau, dtype=float) / N
    return int(np.sum(S[1:RATE_BINS + 1] >= L2_MIN_SURVIVAL)) - 1


def rate_l2(lambda_hat_b, lambda_b, b_max):
    """L2 = sqrt(sum_{b=0}^{b_max} (lambda_hat_b - lambda_b)^2), the L2 distance between the inferred and the
    simulated coalescence rates over bins 0 to b_max, unweighted (deng2025general, SI Appendix, Section 2G). NaN when
    b_max = -1."""
    if b_max < 0:
        return np.nan
    return float(np.sqrt(np.sum(np.square(lambda_hat_b[:b_max + 1] - lambda_b[:b_max + 1]))))


def window_starts(simulated_ts):
    """The starts of the 10 kb windows [10^4 (w - 1), 10^4 w) that lie inside [s_min, s_max], in base pairs."""
    sites = simulated_ts.tables.sites.position
    first = np.ceil(sites[0] / WINDOW_SIZE) * WINDOW_SIZE
    last = sites[-1] // WINDOW_SIZE * WINDOW_SIZE
    return np.arange(first, last, WINDOW_SIZE)


def tskit_windows(ts, starts):
    """The window edges for tskit, from 0 to the sequence length of ts, whose inner windows are the 10 kb windows of
    starts, and the indices of those inner windows."""
    inner_edges = np.append(starts, starts[-1] + WINDOW_SIZE)
    edges = np.unique(np.concatenate([[0.0], inner_edges, [ts.sequence_length]]))
    first = int(np.searchsorted(edges, inner_edges[0]))
    return edges, np.arange(first, first + len(starts))


def window_statistics(ts, starts, mu):
    """pi_w and D_w of each 10 kb window: mu times the branch-mode diversity and segregating sites of tskit, which are
    the means over the window of 2 t_ij over the pairs and of the total length of the branches that subtend 1 to
    n - 1 samples, so pi_w = mu sum_T (l_Tw / 10^4) (1/P) sum_{i<j} 2 t_ij(T) and D_w = mu sum_T (l_Tw / 10^4) L(T),
    with l_Tw the length of window w that tree T covers."""
    edges, inner = tskit_windows(ts, starts)
    pi_w = mu * ts.diversity(mode="branch", windows=edges)[inner]
    D_w = mu * ts.segregating_sites(mode="branch", windows=edges)[inner]
    return pi_w, D_w


def window_errors(estimate, simulated):
    """W_prime, RMSLE, and bias of the log ratios e_w = ln(estimate_w) - ln(simulated_w) of a window statistic over the
    W_prime windows where both values are positive. The log ratio has no 1 added, since the statistics are near
    10^-4, where ln(1 + pi) would turn the ratio into a difference."""
    both = (estimate > 0) & (simulated > 0)
    summary = error_summary(np.log(estimate[both]) - np.log(simulated[both]))
    return {"W_prime": summary["N"], "RMSLE": summary["RMSLE"], "bias": summary["bias"]}


def mutation_ages(ts):
    """{site key: age} for each site of ts with exactly one mutation, the age a = (t_child + t_parent)/2 being the
    midpoint of the times of the nodes of the mutation's edge (pope2026tracing). The key is the position with position
    0 moved to 1, as in the VCF that Relate and SINGER read (np.fmax(1, x)), so that the same site matches across
    tools. A mutation above a root has no edge and is left out."""
    tables = ts.tables
    per_site = np.bincount(tables.mutations.site, minlength=ts.num_sites)
    single = per_site[tables.mutations.site] == 1
    edge = ts.mutations_edge
    keep = single & (edge >= 0)
    child = tables.mutations.node[keep]
    parent = tables.edges.parent[edge[keep]]
    age = (ts.nodes_time[child] + ts.nodes_time[parent]) / 2
    keys = np.fmax(1, tables.sites.position[tables.mutations.site[keep]])
    return dict(zip(keys.tolist(), age.tolist()))


def mutation_age_scores(a, a_hat):
    """N_a, RMSLE, bias, r, and rho of the mutation ages over the N_a sites with a simulated age a_s (dictionary a)
    and an inferred age a_hat_s (dictionary a_hat), with e_s = ln(1 + a_hat_s) - ln(1 + a_s), and the 2D histogram of
    (a_s, a_hat_s) on the bins of TAU (deng2025robust, Fig. 3a). r and rho are scipy.stats.pearsonr and spearmanr of
    ln(1 + a_hat_s) and ln(1 + a_s) over the sites, NaN with fewer than three sites or with constant ages
    (wohns2022unified)."""
    keys = sorted(set(a) & set(a_hat))
    a_s = np.array([a[key] for key in keys], dtype=float)
    a_hat_s = np.array([a_hat[key] for key in keys], dtype=float)
    summary = error_summary(log_errors(a_hat_s, a_s))
    scores = {"N_a": summary["N"], "RMSLE": summary["RMSLE"], "bias": summary["bias"], "r": np.nan, "rho": np.nan}
    if len(keys) > 2 and np.ptp(a_hat_s) > 0 and np.ptp(a_s) > 0:
        scores["r"] = float(stats.pearsonr(np.log1p(a_hat_s), np.log1p(a_s)).statistic)
        scores["rho"] = float(stats.spearmanr(np.log1p(a_hat_s), np.log1p(a_s)).statistic)
    return scores, histogram_2d_counts(a_s, a_hat_s)


def kl_from_uniform(f_r):
    """KL = sum over f_r > 0 of f_r ln(f_r (M + 1)), the Kullback-Leibler divergence of the rank fractions f_r,
    r = 0 to M, from the uniform distribution on 0 to M (yCBrandt2022evaluation), as scipy.stats.entropy."""
    return float(stats.entropy(f_r, np.full(len(f_r), 1 / len(f_r))))


def interval_bounds(samples, axis):
    """{gamma: (lower, upper)}, the (1 - gamma)/2 and (1 + gamma)/2 quantiles of the samples along axis for each
    level gamma, with numpy.quantile's default linear method. One numpy.quantile call gives the six quantiles."""
    probabilities = [p for gamma in GAMMA for p in ((1 - gamma) / 2, (1 + gamma) / 2)]
    quantiles = np.quantile(samples, probabilities, axis=axis)
    return {gamma: (quantiles[2 * index], quantiles[2 * index + 1]) for index, gamma in enumerate(GAMMA)}


def sample_ranks_and_coverage(chunks):
    """rank_counts, the number of simulated values with rank R = r for r = 0 to M, covered, {gamma: the number of
    simulated values inside the central interval of level gamma, ends included}, and N, the number of simulated
    values, for chunks of (samples [M, k, P], t [k, P]) over consecutive positions. The rank R_ijk is the number of
    samples strictly below t_ij(x_k) (yCBrandt2022evaluation), and f_r = rank_counts / N. M samples from a calibrated
    posterior cover about (M - 1) gamma/(M + 1) of the simulated values: a new draw falls with equal probability into
    each of the M + 1 gaps around the sorted samples, and the interval spans (M - 1) gamma of them."""
    rank_counts, covered, N = None, {gamma: 0 for gamma in GAMMA}, 0
    for samples, t in chunks:
        R = (samples < t[None]).sum(axis=0)
        counts = np.bincount(R.ravel(), minlength=samples.shape[0] + 1)
        rank_counts = counts if rank_counts is None else rank_counts + counts
        for gamma, (lower, upper) in interval_bounds(samples, 0).items():
            covered[gamma] += int(((lower <= t) & (t <= upper)).sum())
        N += t.size
    return rank_counts, covered, N


def draw_ranks_and_coverage(draws, mrca, t):
    """rank_counts, covered, and N of sample_ranks_and_coverage for draws on one topology: draws [nodes, M] are the
    posterior draws of every node's time, and the samples of a simulated value t [K, P] are the M draws of its MRCA
    node mrca [K, P]. The values with the same MRCA node have the same samples, so the interval bounds are computed
    once per node."""
    M = draws.shape[1]
    rank_counts = np.zeros(M + 1, dtype=np.int64)
    for start in range(0, len(t), CHUNK):
        chunk = slice(start, start + CHUNK)
        # Number of draws of the MRCA node strictly below the simulated value.
        R = (draws[mrca[chunk]] < t[chunk][..., None]).sum(axis=-1)
        rank_counts += np.bincount(R.ravel(), minlength=M + 1)
    # Interval bounds per node, then per value through its MRCA node.
    covered = {gamma: int(((lower[mrca] <= t) & (t <= upper[mrca])).sum())
               for gamma, (lower, upper) in interval_bounds(draws, 1).items()}
    return rank_counts, covered, t.size


def binomial_rank_fractions(U, M=100):
    """f_r = (1/N) sum C(M, r) U^r (1 - U)^(M - r) for r = 0 to M, over the N values of U = F(t), the cumulative
    distribution function of each value's posterior at the simulated value: the expected rank fractions of M
    independent draws from each posterior. The binomial probabilities are written out with scipy.special.comb,
    because scipy.stats.binom.pmf raises an OverflowError at some U below 1e-300, and numpy's 0^0 = 1 gives rank 0 at
    U = 0 and rank M at U = 1. The values that share a U are computed once, weighted by their number."""
    r = np.arange(M + 1)
    distinct, counts = np.unique(np.clip(np.ravel(U), 0.0, 1.0), return_counts=True)
    total = np.zeros(M + 1)
    for start in range(0, len(distinct), 100_000):
        u = distinct[start:start + 100_000, None]
        # Probability of each rank at each distinct U, summed with the number of values with that U as weights.
        probabilities = special.comb(M, r) * u ** r * (1 - u) ** (M - r)
        total += counts[start:start + 100_000] @ probabilities
    return total / np.size(U)


def gamma_posteriors(mean, variance):
    """The shape mean^2/variance and the rate mean/variance of each node's gamma posterior, from tsdate's posterior
    mean and variance (pope2026tracing). NaN for the nodes with variance 0, the samples."""
    with np.errstate(divide="ignore", invalid="ignore"):
        shape, rate = np.square(mean) / variance, mean / variance
    fixed = ~(variance > 0)
    shape[fixed], rate[fixed] = np.nan, np.nan
    return shape, rate


def gamma_calibration(shape, rate, mrca, t):
    """U = F(t) [K, P], the cumulative distribution function of the gamma posterior of the MRCA node at each
    simulated value, and covered, {gamma: the number of simulated values inside [F^-1((1 - gamma)/2),
    F^-1((1 + gamma)/2)]}, with the quantiles computed once per node."""
    U = stats.gamma.cdf(t, shape[mrca], scale=1 / rate[mrca])
    covered = {}
    for gamma in GAMMA:
        # Quantiles per node, then per value through its MRCA node.
        lower = stats.gamma.ppf((1 - gamma) / 2, shape, scale=1 / rate)
        upper = stats.gamma.ppf((1 + gamma) / 2, shape, scale=1 / rate)
        covered[gamma] = int(((lower[mrca] <= t) & (t <= upper[mrca])).sum())
    return U, covered


def triple_pairs(n):
    """[triples, 3] pair indices of the three pairs (i, j), (i, l), and (j, l) of every triple of samples i < j < l,
    among the P pairs."""
    pair_index = np.zeros((n, n), dtype=np.int64)
    first, second = np.triu_indices(n, 1)
    pair_index[first, second] = np.arange(len(first))
    triples = np.array(list(combinations(range(n), 3)))
    return np.stack([pair_index[triples[:, 0], triples[:, 1]], pair_index[triples[:, 0], triples[:, 2]],
                     pair_index[triples[:, 1], triples[:, 2]]], axis=1)


def triple_labels(clade_size, pairs):
    """[k, triples] labels (int8) of every triple at each position, from the clade sizes [k, P] of the pairs: 0, 1,
    or 2 for the pair (i, j), (i, l), or (j, l) whose MRCA has strictly the fewest samples below it, or 3 for an
    unresolved triple. In a tree, two of the three pairs have the MRCA of the triple as their MRCA, so their clade
    sizes are equal and the third is at most theirs. With a, b, c the clade sizes of (i, j), (i, l), (j, l), the label
    is 0 if a < b, else 1 if b < a, else 2 if c < a, else 3, three comparisons in place of six."""
    a, b, c = (clade_size[:, pairs[:, column]] for column in range(3))
    # At most one pair has its MRCA strictly below the MRCA of the triple.
    return np.where(a < b, 0, np.where(b < a, 1, np.where(c < a, 2, 3))).astype(np.int8)


def wrong_triplets(simulated_labels, clade_size_hat, pairs):
    """WT, the mean over positions of WT_k, the fraction of the C(n, 3) triples whose label in the inferred tree
    differs from the label in the simulated tree (the triplet distance of deng2025robust), from the simulated labels
    [K, triples] (triple_labels) and the inferred clade sizes [K, P]. The inferred labels come from the tree before
    any random split, so an unresolved triple counts as wrong, since every simulated triple is resolved."""
    total = 0.0
    for start in range(0, len(simulated_labels), CHUNK):
        chunk = slice(start, start + CHUNK)
        different = simulated_labels[chunk] != triple_labels(clade_size_hat[chunk], pairs)
        total += different.mean(axis=1).sum()
    return total / len(simulated_labels)


def tree_distances(ts, simulated_ts, x):
    """RF, the mean over positions of RF_k, and KC_sum, the sum over positions of KC(T_hat_star_k, T_k), for the
    inferred tree sequence ts against the simulation, simulated_ts. T_hat_star_k is the tree of ts at x_k with every
    polytomy split into a binary subtree drawn with equal probability among all binary trees on its children (tskit
    Tree.split_polytomies with random seed k, wohns2022unified), or the tree itself when it has no polytomy.
    RF_k = (|C(T) - C(T_hat_star)| + |C(T_hat_star) - C(T)|)/(2n - 4) is the rooted distance of tskit
    Tree.rf_distance divided by its maximum for two binary trees (zhang2023biobank-scale), and
    KC = sqrt(sum_{i<j} (delta_ij(T_1) - delta_ij(T_2))^2) is tskit Tree.kc_distance at lambda 0, delta_ij being the
    number of edges from the root to the MRCA of i and j. ts is first simplified without its sites, which removes its
    unary nodes, as kc_distance requires, and keeps the samples as nodes 0 to n - 1 and the clades. The sites go
    first, since nested mutations in SINGER's samples can block simplify."""
    n = ts.num_samples
    simplified = ts.delete_sites(np.arange(ts.num_sites)).simplify()
    T_hat, T = tskit.Tree(simplified, sample_lists=True), tskit.Tree(simulated_ts, sample_lists=True)
    RF_sum = KC_sum = 0.0
    for k, x_k in enumerate(x, start=1):
        T_hat.seek(x_k)
        T.seek(x_k)
        # A tree with a polytomy is split at random, and a binary tree is compared as it is. The last entry of the
        # array is the virtual root, whose children are the roots.
        if T_hat.num_children_array[:-1].max() > 2:
            T_hat_star = T_hat.split_polytomies(random_seed=k, record_provenance=False, sample_lists=True)
        else:
            T_hat_star = T_hat
        RF_sum += T_hat_star.rf_distance(T)
        KC_sum += T_hat_star.kc_distance(T, 0)
    return {"RF": RF_sum / len(x) / (2 * n - 4), "KC_sum": KC_sum}


def random_tree_kc(simulated_ts, x):
    """KC_rand_sum, the sum over positions of KC(T_rand_k, T_k), the denominator of the KC metric
    KC = sum_k KC(T_hat_star_k, T_k) / sum_k KC(T_rand_k, T_k) (wohns2022unified). T_rand_k is a binary tree with n
    leaves drawn with equal probability among all such trees (tskit Tree.generate_random_binary with random seed k).
    It depends only on the simulation, so every configuration of a data set divides by the same value."""
    n = simulated_ts.num_samples
    T = tskit.Tree(simulated_ts, sample_lists=True)
    KC_rand_sum = 0.0
    for k, x_k in enumerate(x, start=1):
        T.seek(x_k)
        T_rand = tskit.Tree.generate_random_binary(n, random_seed=k, record_provenance=False, sample_lists=True)
        KC_rand_sum += T_rand.kc_distance(T, 0)
    return KC_rand_sum
