"""
time_predictions.py: argGNN's prediction time of the testing data sets, in one run for every checkpoint
=======================================================================================================
Not a pipeline rule. For each testing file of the command line, it predicts argGNN's clade-size matrices with each
checkpoint of the command line, with score_arggnn.predict, and times the prediction alone, as
score_arggnn.kept_predictions times it. With two checkpoints, the two predictions of a file run one after the
other in the same job, so their times compare. Every prediction is compared with the one kept in
OUT_DIR/arggnn_predictions/{model}/{tag}.npz, which the benchmark used, and nothing is written there.

Why: the benchmark of the paper read the predictions that an earlier run made and kept, and not every job of that run
finished, so 448 of the 5,000 data sets of the base checkpoint had no prediction time in any table. One run of this
script timed every prediction again, those of both checkpoints, on one CPU, and the figures of the paper take
argGNN's prediction time from its tables.

Outputs, for each checkpoint, in its OUT_DIR:
  prediction_times/{model}/{model}_S{size}_{first}_to_{last}.csv   one row per testing file of the replicates first to
      last: model, size, replicate, wall_seconds_predict, identical_to_kept (the prediction equals the kept one),
      max_abs_difference (the largest absolute difference to the kept one, NaN without a kept one of the same shape),
      node, and checkpoint_md5
A table is written once all its testing files are done, so a job that stops leaves no partial table.

Run from the repo root, with PYTHONPATH=src, and each OUT_DIR the output folder of src/score_arggnn.py for its
checkpoint, benchmarking/<name>/<prefix>:
  python -u paper_figures/time_predictions.py CHECKPOINT_DIR[,CHECKPOINT_DIR] OUT_DIR[,OUT_DIR] \
      TRUE_FILE [TRUE_FILE ...]
"""
import os
import socket
import sys
import time

import numpy as np
import pandas as pd
import yaml

import score_arggnn as SA
import run_external_tools as R

COLUMNS = ["model", "size", "replicate", "wall_seconds_predict", "identical_to_kept", "max_abs_difference", "node",
           "checkpoint_md5"]


def settings_of(checkpoint_dir):
    """The GNN_* settings of the shared and benchmark blocks of a checkpoint's config, as
    score_arggnn.import_arggnn reads them.

    Args:
        checkpoint_dir (str): Folder of the checkpoint, with {name}.pt and {name}.yaml.

    Returns:
        settings (dict): The settings, as strings.
    """
    name = os.path.basename(os.path.normpath(checkpoint_dir))
    with open(os.path.join(checkpoint_dir, f"{name}.yaml")) as fh:
        config = yaml.safe_load(fh)
    return {key: str(value) for block in ("shared", "benchmark") for key, value in config[block].items()}


def compare_with_kept(path, matrices, positions):
    """Whether a prediction equals the kept one, and their largest absolute difference.

    Args:
        path (str): The kept predictions of the testing file.
        matrices (numpy.ndarray): SNPs x pairs, float16, as score_arggnn.predict returns them.
        positions (numpy.ndarray): The position of each SNP.

    Returns:
        identical (bool): The matrices and positions equal the kept ones.
        difference (float): The largest absolute difference of the matrices, NaN without a kept file of the same
            shape and positions.
    """
    if not os.path.exists(path):
        return False, float("nan")
    kept = np.load(path)
    if kept["matrices"].shape != matrices.shape or not np.array_equal(kept["positions"], positions):
        return False, float("nan")
    difference = np.abs(kept["matrices"].astype(np.float32) - matrices.astype(np.float32))
    return bool(np.array_equal(kept["matrices"], matrices)), float(difference.max()) if difference.size else 0.0


def main():
    """Time the prediction of every testing file of the command line with every checkpoint, and write the tables."""
    if len(sys.argv) < 4:
        raise SystemExit("!! usage: python -u paper_figures/time_predictions.py "
                         "CHECKPOINT_DIR[,CHECKPOINT_DIR] OUT_DIR[,OUT_DIR] TRUE_FILE [TRUE_FILE ...]")
    checkpoint_dirs, out_dirs, true_files = sys.argv[1].split(","), sys.argv[2].split(","), sys.argv[3:]
    if len(checkpoint_dirs) != len(out_dirs):
        raise SystemExit(f"!! {len(checkpoint_dirs)} checkpoints but {len(out_dirs)} output folders")
    # argGNN's modules read their settings once, at import, so every checkpoint must have the settings of the first.
    for checkpoint_dir in checkpoint_dirs[1:]:
        if settings_of(checkpoint_dir) != settings_of(checkpoint_dirs[0]):
            raise SystemExit(f"!! {checkpoint_dir} has other settings than {checkpoint_dirs[0]}")
    settings, _ = SA.import_arggnn(checkpoint_dirs[0])
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node = socket.gethostname()
    print(f"time_predictions | {len(true_files)} testing files | device {device} | node {node} | torch threads "
          f"{torch.get_num_threads()}", flush=True)
    print("  " + " ".join(f"{key}={value}" for key, value in sorted(settings.items())), flush=True)
    models = []
    for checkpoint_dir in checkpoint_dirs:
        name = os.path.basename(os.path.normpath(checkpoint_dir))
        checkpoint = os.path.join(checkpoint_dir, f"{name}.pt")
        md5 = R.md5_of_file(checkpoint)
        models.append((name, SA.load_model(checkpoint, device), md5))
        print(f"  checkpoint {R.relative(os.path.abspath(checkpoint))} md5 {md5}", flush=True)
    rows = [[] for _ in checkpoint_dirs]
    for true_file in true_files:
        mts, row, tag = R.read_simulation(true_file)
        # The predictions of the checkpoints run one after the other, so that their times compare.
        for k, ((name, model, md5), out_dir) in enumerate(zip(models, out_dirs)):
            start = time.perf_counter()
            matrices, positions = SA.predict(model, mts, device)
            seconds = round(time.perf_counter() - start, 2)
            kept = os.path.join(out_dir, "arggnn_predictions", row["model"], f"{tag}.npz")
            identical, difference = compare_with_kept(kept, matrices, positions)
            rows[k].append({"model": row["model"], "size": mts.num_samples, "replicate": int(row["replicate"]),
                            "wall_seconds_predict": seconds, "identical_to_kept": identical,
                            "max_abs_difference": difference, "node": node, "checkpoint_md5": md5})
            print(f"  {tag} | {name} | {seconds} s | identical to the kept prediction {identical}, largest absolute "
                  f"difference {difference}", flush=True)
    for rows_of_checkpoint, out_dir in zip(rows, out_dirs):
        table_df = pd.DataFrame(rows_of_checkpoint, columns=COLUMNS)
        # One table per model and size, named by its first and last replicate.
        for (model, size), group_df in table_df.groupby(["model", "size"]):
            path = os.path.join(out_dir, "prediction_times", model,
                                f"{model}_S{size}_{group_df['replicate'].min()}_to_{group_df['replicate'].max()}.csv")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # The table goes through a temporary name, so that an interrupted job leaves no table.
            group_df.to_csv(path + ".tmp", index=False)
            os.replace(path + ".tmp", path)
            print(f"  {R.relative(os.path.abspath(path))}: {len(group_df)} rows", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
