# ARGNN

A deep learning method for inferring ancestral recombination graphs.

## Repository layout

```
Snakefile                          the Snakemake pipeline
config/
  const_split_bott_zigzag_1s14__seq256_knn7w_rank3.yaml   the config used
hpc/
  profile/config.yaml              the SLURM profile for the Snakemake pipeline
  scripts/venv_setup.sh            builds .venv
  scripts/bin_downloads.sh         installs Relate, SINGER, and POLEGON in bin/
  scripts/snakemake_driver.sh      runs the Snakemake pipeline as a SLURM job
src/
  simulate_data.py                 simulates the training, testing, and fine-tuning data
  run_external_tools.py            runs Relate, tsinfer+tsdate, and SINGER+POLEGON
  model.py                         the GNN architecture
  gnn_topology.py                  the settings, losses, metrics, and decoders
  train.py                         trains a checkpoint
  finetune.py                      fine-tunes a checkpoint
  branch_length.py                 assembles the decoded trees into a tree sequence and dates it
  export_trees.py                  writes ARGNN's dated tree sequence from a simulation
  vcf_to_trees.py                  writes ARGNN's dated tree sequence from a phased VCF
  arg_metrics.py                   the benchmarking metrics
  score_benchmark.py               scores the inferred ARGs from an external tool
  score_arggnn.py                  infers, dates, and scores ARGNN's ARGs
checkpoints/                       the base and two fine-tuned checkpoints
benchmarking/const_split_bott_ooa_zigzag_1s14__relate_tsinfer_true_topology_singer/
  <checkpoint>/prediction_times/   the prediction times of the checkpoints
paper_figures/                     the notebooks, summary tables, and figures
archaic_introgression/             archaic introgression analyses
requirements*.txt                  the dependencies
```

The Snakemake pipeline writes `training_data/`, `testing_data/`, `external_data/`, and the score tables of `benchmarking/`, and the setup writes `.venv/` and `bin/`.

## Setup

    bash hpc/scripts/venv_setup.sh      # builds .venv from requirements-frozen-oscar.txt
    bash hpc/scripts/bin_downloads.sh   # installs Relate, SINGER, and POLEGON in bin/
    mkdir -p hpc/logs                   # builds the logs directory for SLURM


The Snakemake pipeline was run on [OSCAR](https://docs.ccv.brown.edu/oscar). To re-run on your own cluster, and assuming your cluster uses SLURM, you will need to change:

- the Python module, `module load python/3.13.5-dz6t`, in `hpc/scripts/venv_setup.sh` (line 14) and `hpc/scripts/snakemake_driver.sh` (line 39);
- the partitions in `hpc/profile/config.yaml`: `batch` for every job (line 19) and `gpu` for training (line 25);
- the GPU types of training in `hpc/profile/config.yaml` (line 27).

## Running the pipeline

The Snakemake pipeline has five stages:

| Stage | Script | Writes |
| --- | --- | --- |
| `training_data` | `src/simulate_data.py` | `training_data/<model>/` |
| `testing_data` | `src/simulate_data.py` | `testing_data/<model>/` |
| `external_data` | `src/run_external_tools.py` | `external_data/<tool>/testing_data/<model>/` |
| `train` | `src/train.py` | `checkpoints/<prefix>/` |
| `benchmark` | `src/score_benchmark.py`, `src/score_arggnn.py` | `benchmarking/<name>/scores/<tool>/<model>/`, `benchmarking/<name>/<prefix>/` |

`hpc/scripts/snakemake_driver.sh` runs the Snakemake pipeline as a SLURM job, and Snakemake submits one SLURM job per step with the profile `hpc/profile/config.yaml`. Its arguments are the config, a stage, and options for Snakemake. Include the `-n` flag after the stage to submit a dry run. To reproduce the results generated here, run:

    CONFIG=config/const_split_bott_zigzag_1s14__seq256_knn7w_rank3.yaml
    sbatch hpc/scripts/snakemake_driver.sh $CONFIG training_data
    sbatch hpc/scripts/snakemake_driver.sh $CONFIG testing_data
    sbatch hpc/scripts/snakemake_driver.sh $CONFIG external_data
    sbatch hpc/scripts/snakemake_driver.sh $CONFIG train
    sbatch hpc/scripts/snakemake_driver.sh $CONFIG benchmark

Start each stage after the driver of the previous stage has finished, and the corresponding log of the Snakemake driver will output to `hpc/logs/argGNN_snakemake_<job ID>.err`.

The run times of the profile are for a small test run, but `--set-resources` replaces the whole `set-resources` block of the profile. The paper's training ran with:

    sbatch -t 7-00:00:00 hpc/scripts/snakemake_driver.sh $CONFIG train \
        --set-resources train_model:slurm_partition=gpu train_model:gres=gpu:1 train_model:constraint=l40s \
        train_model:cpus_per_task=8 train_model:mem_mb=196608 train_model:runtime=3600

Note that `sbatch -t` replaces the driver's default runtime of 24 hours. A full run also needs longer runtimes, e.g., for SINGER and `score_arggnn`, `singer:runtime=300` and `score_arggnn:runtime=1440` (in minutes).

## The fine-tuned checkpoints

`src/finetune.py` trains the base checkpoint on additional simulated data from a single demographic model. It holds out 20% of the simulations and keeps the epoch with the lowest held-out loss. The two fine-tuned checkpoints ran 10 additional epochs at a learning rate of 2e-4, respectively, for the following demographic models:

- `ooa` (`stdpopsim`'s `OutOfAfricaArchaicAdmixture_5R19`): 24 replicates of 20 Mb at 10, 20, and 50 haplotypes;
- `papuans_10j19` (`stdpopsim`'s `PapuansOutOfAfrica_10J19`): 24 replicates of 20 Mb at 52 haplotypes: 24 Papuan, 24 Yoruba, two Neanderthal, and two Denisovan.

The simulated data for fine-tuning comes from the `training_data` block of the config, by running:

    sbatch hpc/scripts/snakemake_driver.sh $CONFIG \
        training_data/ooa/ooa_S10_simulations.csv training_data/ooa/ooa_S20_simulations.csv \
        training_data/ooa/ooa_S50_simulations.csv training_data/papuans_10j19/papuans_10j19_S52_simulations.csv

The fine-tuning needs the `shared` and `train` blocks of the config as environment variables. For example, to run a fine-tuning job for `ooa` with one GPU, run:

    source .venv/bin/activate
    C=const_split_bott_zigzag_1s14__seq256_knn7w_rank3
    F=${C}_finetuned_ooa_10_epochs_lr_2e-4
    eval "$(python -P -c 'import shlex, sys, yaml; c = yaml.safe_load(open(sys.argv[1])); print("\n".join(f"export {k}={shlex.quote(str(v))}" for b in ("shared", "train") for k, v in c[b].items()))' config/$C.yaml)"
    mkdir -p checkpoints/$F
    python -u src/finetune.py --ckpt checkpoints/$C/$C.pt --data-dir training_data/ooa --out checkpoints/$F/$F.pt \
        --epochs 10 --lr 2e-4 2>&1 | tee checkpoints/$F/finetune.log

Each fine-tuned directory contains the checkpoint, `finetune.log`, and the config of the base checkpoint under a header that names the fine-tuning.

## Training and benchmarking results

The `paper_figures/` directory contains the training and benchmarking results and the following analysis notebooks:

- `benchmark_summary_tables.ipynb` reads the score tables and writes the two summary tables in `benchmark_summary_tables/`. The score tables are not in the repository as the files are too large, but the `benchmark` stage writes them. The `benchmark` stage scores only the checkpoint specified in the config. The tables for the fine-tuned `ooa` checkpoint were generated with:

      N=benchmarking/const_split_bott_ooa_zigzag_1s14__relate_tsinfer_true_topology_singer
      F=const_split_bott_zigzag_1s14__seq256_knn7w_rank3_finetuned_ooa_10_epochs_lr_2e-4
      OMP_NUM_THREADS=1 python -u src/score_arggnn.py checkpoints/$F $N/$F testing_data/ooa/*.trees

- `benchmark_figures.ipynb` exports the main and supplementary plots to `figures/` from the two summary tables and the prediction times.
- `training_curves.ipynb` exports `figures/training_loss.pdf` and `figures/finetuning_loss.pdf` from `train.log` and the two `finetune.log` files.
