"""
score_arggnn.py: argGNN's trees of testing files, each dated by tsdate and by POLEGON, and their benchmark metrics
==================================================================================================================
For each testing file of src/simulate_data.py, it predicts argGNN's clade-size matrix at every SNP with a checkpoint,
decodes the avg_constr tree of every SNP and assembles the trees with branch_length.assemble_decoded, dates the
assembly with tsdate and, separately, with POLEGON, and scores both with the metrics of src/score_benchmark.py.

Configurations:
  arggnn_tsdate   the assembly dated by tsdate as run_external_tools.date_with_tsdate dates it (variational_gamma,
                  match_segregating_sites=True), scored as score_benchmark.score_tsdate scores tsinfer
  arggnn_polegon  the assembly dated by POLEGON as run_external_tools.polegon_ages dates it (posterior means, from the
                  placeholder times of the assembly, the clade size of each node), scored as
                  score_benchmark.score_true_topology scores true_topology_polegon

Settings: the shared and benchmark blocks of the config that the Snakefile leaves next to the checkpoint go into the
environment before train.py and branch_length.py are imported, as the Snakefile gives them to export_trees.py. The
settings of the benchmark, those of DEFAULTS, are the defaults: the avg_constr decoder, the bridge_noroot assembly with
EPS 1 and a root tolerance of 1 clade switch from the first split of each run, and each change of tree at the
midpoint between two SNPs. The log prints the settings that branch_length.py holds.

Prediction: the steps of export_trees.model_to_tree_sequence up to the decode. The model comes from the shapes of the
checkpoint, the haplotype window is checked, and train._sliding_predict runs on train._flat_stream under
torch.no_grad(). The matrices are kept as float16 upper triangles, and the decoder reads the kept values, so a rerun
from a kept file gives the same trees.

Rows, those of score_benchmark.py for each configuration, plus num_nodes, num_edges, num_sites, and num_mutations of
the assembly, and wall_seconds_{step} for predict (when the matrices were predicted in this run), decode (the decode
of every SNP and the counts of p_k), assemble (the assembly, the mutations, and the edge squash), and tsdate or
polegon.

Outputs, in OUT_DIR:
  scores/{model}/{model}_S{size}_{replicate}.csv.gz   the table of score_benchmark.write_table
  arggnn_predictions/{model}/{model}_S{size}_{replicate}.npz   matrices (SNPs x pairs in np.triu_indices order,
                                                      float16) and positions
  arggnn_trees/{model}/{model}_S{size}_{replicate}/{configuration}.trees   the dated tree sequences, for the replicates
                                                      of GNN_SCORE_KEEP_TREES only
A testing file whose table exists is skipped, and a kept prediction is read in place of a new one. A testing file that
fails is reported and the next one is run, and the script then ends with an error that lists the failures.

Run with, from the repo root:
  python -u src/score_arggnn.py CHECKPOINT_DIR OUT_DIR TRUE_FILE [TRUE_FILE ...]
CHECKPOINT_DIR holds {name}.pt and {name}.yaml, the copy of its config. Env: GNN_SCORE_KEEP_TREES (the replicates
whose dated tree sequences are kept, a comma list, default 0), and GNN_POLEGON_DIR (see run_external_tools.py).
"""

import os
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np
import yaml

import run_external_tools as R
import score_benchmark as SB

# The modules of argGNN, imported by import_arggnn once the GNN_* settings are in the environment.
BL = E = V = None
# The settings of the benchmark, which the config of the checkpoint can change.
DEFAULTS = {"GNN_DECODER": "avg_constr", "GNN_BL_ASSEMBLY": "bridge_noroot", "GNN_BL_EPS": "1.0",
            "GNN_BL_ROOT_TOL": "1", "GNN_BL_ROOT_METRIC": "clade", "GNN_BL_ROOT_CMP": "anchor",
            "GNN_BL_BREAKPOINTS": "midpoint"}
CONFIGURATIONS = ("arggnn_tsdate", "arggnn_polegon")
KEEP_TREES = [int(x) for x in os.environ.get("GNN_SCORE_KEEP_TREES", "0").split(",") if x]


def import_arggnn(checkpoint_dir):
    """Put DEFAULTS and then the shared and benchmark blocks of the config of the checkpoint in the environment, and
    import argGNN's modules. train.py, gnn_topology.py, and branch_length.py read their settings at import, so the
    settings must be in the environment first. Returns the settings of the two blocks and the path of the checkpoint."""
    global BL, E, V
    name = os.path.basename(os.path.normpath(checkpoint_dir))
    checkpoint = os.path.join(checkpoint_dir, f"{name}.pt")
    with open(os.path.join(checkpoint_dir, f"{name}.yaml")) as fh:
        config = yaml.safe_load(fh)
    settings = {key: str(value) for block in ("shared", "benchmark") for key, value in config[block].items()}
    for key, value in DEFAULTS.items():
        os.environ.setdefault(key, value)
    os.environ.update(settings)
    import branch_length as BL
    import export_trees as E
    import train as V
    return settings, checkpoint


def load_model(checkpoint, device):
    """The model of a checkpoint, as export_trees.main builds it, in evaluation mode."""
    import torch
    model, node_feat = E._build_model_from_ckpt(torch.load(checkpoint, map_location=device), device)
    # The checkpoint stores no settings, but its input width is the haplotype window it was trained with.
    if node_feat != V.HAP_W:
        raise SystemExit(f"!! checkpoint hap window W={node_feat} differs from GNN_HAP_WINDOW ({V.HAP_W})")
    return model


def predict(model, mts, device):
    """argGNN's clade-size matrix at every SNP of a simulation, as float16 upper triangles in np.triu_indices order
    (SNPs x pairs), and the position of each SNP."""
    import torch
    S = mts.num_samples
    items = V._flat_stream(mts, S)
    if not items:
        raise ValueError("no SNPs in the input sequence")
    with torch.no_grad():
        preds, _ = V._sliding_predict(model, items, S, device)
    upper = np.triu_indices(S, 1)
    matrices = np.stack([np.asarray(p)[upper] for p in preds]).astype(np.float16)
    return matrices, np.array([it[2] for it in items])


def kept_predictions(path, mts, model, device):
    """The kept predictions of a simulation, made and kept first if the file does not exist. Returns the matrices, as
    predict returns them, and the wall time of the prediction, or None if the file was read."""
    seconds = None
    if os.path.exists(path):
        kept = np.load(path)
        matrices, positions = kept["matrices"], kept["positions"]
    else:
        start = time.perf_counter()
        matrices, positions = predict(model, mts, device)
        seconds = round(time.perf_counter() - start, 2)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # The file goes through a temporary name that ends in .npz, since numpy adds that ending to any other name,
        # so that an interrupted job leaves no file.
        temporary = path[:-len(".npz")] + ".tmp.npz"
        np.savez_compressed(temporary, matrices=matrices, positions=positions)
        os.replace(temporary, path)
    S = mts.num_samples
    if not np.array_equal(positions, mts.tables.sites.position) or matrices.shape[1] != S * (S - 1) // 2:
        raise SystemExit(f"!! {R.relative(path)} does not hold one matrix of {S} haplotypes per site of the "
                         f"simulation")
    return matrices, seconds


def square_matrices(matrices, num_samples):
    """The kept upper triangles as symmetric num_samples x num_samples matrices, with 0 on the diagonal."""
    upper = np.triu_indices(num_samples, 1)
    squares = []
    for row in matrices:
        square = np.zeros((num_samples, num_samples))
        square[upper] = row.astype(np.float64)
        squares.append(square + square.T)
    return squares


def topology_rows(configuration, ts, values):
    """The rows of one configuration that describe its topology and the steps before the dating, from (quantity,
    value) pairs."""
    rows = []
    counts = [("num_nodes", ts.num_nodes), ("num_edges", ts.num_edges), ("num_sites", ts.num_sites),
              ("num_mutations", ts.num_mutations)]
    for quantity, value in counts + values:
        SB.add(rows, configuration, quantity, value)
    return rows


def polegon_rows(configuration, ts, mean, draws, simulated):
    """The rows of a topology dated by POLEGON's posterior means, with the calibration by POLEGON's draws (one row per
    node of ts and one column per draw) of the MRCA node of each simulated value, as score_benchmark.
    score_true_topology scores true_topology_polegon."""
    accumulator = SB.new_accumulator(simulated)
    mrca = SB.add_tree_sequence(accumulator, ts, SB.polegon_dated(ts, mean), simulated)
    rows = SB.configuration_rows(configuration, accumulator, simulated)
    rank_counts, covered, N = SB.A.draw_ranks_and_coverage(draws, mrca, simulated["t"])
    return rows + SB.calibration_rows(configuration, rank_counts / N, covered, N, draws.shape[1])


def date_and_score(ts, values, simulated, work, keep_dir):
    """The rows of arggnn_tsdate and arggnn_polegon: the topology ts, with placeholder node times, dated by tsdate and,
    separately, by POLEGON, and both scored. values are the (quantity, value) pairs of the steps before the dating.
    The dated tree sequences are kept in keep_dir unless it is None."""
    steps = []
    R.date_with_tsdate(ts, "arggnn", work, steps)
    configuration = "arggnn_tsdate"
    rows, _, _ = SB.score_tsdate(work, simulated, "arggnn", configuration)
    rows += topology_rows(configuration, ts, values + [("wall_seconds_tsdate", steps[-1]["wall_seconds"])])
    with R.timed_step("polegon", steps):
        mean, draws = R.polegon_ages(ts, work)
    configuration = "arggnn_polegon"
    rows += polegon_rows(configuration, ts, mean, draws, simulated)
    rows += topology_rows(configuration, ts, values + [("wall_seconds_polegon", steps[-1]["wall_seconds"])])
    trees, posteriors = (os.path.join(work, f"arggnn{end}") for end in (".trees", "_node_posteriors.npz"))
    if keep_dir:
        shutil.copyfile(trees, os.path.join(keep_dir, "arggnn_tsdate.trees"))
        SB.polegon_dated(ts, mean).dump(os.path.join(keep_dir, "arggnn_polegon.trees"))
    os.remove(trees)
    os.remove(posteriors)
    return rows


def run_testing_file(true_file, out_dir, model, device):
    """Both configurations of one testing file, from the prediction to the table. Returns the rows, or None if the
    table exists, and the path of the table."""
    mts, row, tag = R.read_simulation(true_file)
    path = os.path.join(out_dir, "scores", row["model"], f"{tag}.csv.gz")
    if os.path.exists(path):
        return None, path
    S, seqlen = mts.num_samples, float(mts.sequence_length)
    print(f"\n=== {tag} | {S} haplotypes, {mts.num_sites} sites, {mts.num_trees} trees", flush=True)
    simulated = SB.simulation_values(mts)
    rows = SB.simulation_rows(simulated, row)
    matrices, predict_seconds = kept_predictions(
        os.path.join(out_dir, "arggnn_predictions", row["model"], f"{tag}.npz"), mts, model, device)
    values = [] if predict_seconds is None else [("wall_seconds_predict", predict_seconds)]
    print(f"  predictions: {'read' if predict_seconds is None else f'{predict_seconds} s'}", flush=True)
    # The items give each SNP its focal alleles, for the decoder and for the mutations, as in export_trees.py.
    items = V._flat_stream(mts, S)
    V.DECODER = "avg_constr"
    assembled, diag, _ = BL.assemble_decoded(square_matrices(matrices, S), items, S, seqlen)
    values += [("wall_seconds_decode", round(diag["decode_s"], 2)),
               ("wall_seconds_assemble", round(diag["assemble_s"] + diag.get("squash_s", 0.0), 2))]
    print(f"  decode {values[-2][1]} s, assemble {values[-1][1]} s | {assembled.num_trees} trees, "
          f"{assembled.num_nodes} nodes", flush=True)
    keep_dir = None
    if int(row["replicate"]) in KEEP_TREES:
        keep_dir = os.path.join(out_dir, "arggnn_trees", row["model"], tag)
        os.makedirs(keep_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix=f"score_arggnn_{tag}_", dir=os.environ.get("TMPDIR"))
    try:
        rows += date_and_score(assembled, values, simulated, work, keep_dir)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    SB.write_table(path, row, rows)
    return rows, path


def main():
    """Score argGNN on the testing files of the command line."""
    if len(sys.argv) < 4:
        raise SystemExit("!! usage: python -u src/score_arggnn.py CHECKPOINT_DIR OUT_DIR TRUE_FILE [TRUE_FILE ...]")
    checkpoint_dir, out_dir, true_files = sys.argv[1], sys.argv[2], sys.argv[3:]
    settings, checkpoint = import_arggnn(checkpoint_dir)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"score_arggnn | {len(true_files)} testing files | device {device} | torch {torch.__version__} threads "
          f"{torch.get_num_threads()} | checkpoint {R.relative(os.path.abspath(checkpoint))} md5 "
          f"{R.md5_of_file(checkpoint)}", flush=True)
    print("  " + " ".join(f"{key}={value}" for key, value in sorted(settings.items())), flush=True)
    print(f"  branch_length: assembly {BL.BL_ASSEMBLY}, EPS {BL.EPS}, root {BL.ROOT_TOL}/{BL.ROOT_METRIC}/"
          f"{BL.ROOT_CMP}, breakpoints {BL.BREAKPOINTS} | trees kept for replicates {KEEP_TREES}", flush=True)
    R.print_versions("true_topology")
    model = load_model(checkpoint, device)
    failed = []
    for true_file in true_files:
        start = time.perf_counter()
        # One testing file that fails must not stop the others, so any error is reported and the next file run.
        try:
            rows, path = run_testing_file(true_file, out_dir, model, device)
        except (Exception, SystemExit) as error:
            failed.append(true_file)
            print(f"!! {true_file} | {type(error).__name__}: {R.relative(str(error))}", flush=True)
            print(R.relative(traceback.format_exc()), flush=True)
            continue
        if rows is None:
            print(f"=== {os.path.basename(path)} | table exists, skipped", flush=True)
            continue
        print(f"  {len(rows)} rows, {os.path.getsize(path) / 1e3:.0f} kB, {time.perf_counter() - start:.0f} s",
              flush=True)
        for configuration in CONFIGURATIONS:
            print(f"    {configuration} | {SB.summary_text(rows, configuration)}", flush=True)
    if failed:
        raise SystemExit(f"!! {len(failed)} of {len(true_files)} testing files failed: {' '.join(failed)}")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
