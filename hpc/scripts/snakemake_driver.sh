#!/bin/bash
# =============================================================================
# snakemake_driver.sh: runs the Snakefile on Oscar, and snakemake submits 1 Slurm job per rule call
# (profile hpc/profile/, job logs in hpc/logs/slurm/). Submit from the repo root:
#   sbatch hpc/scripts/snakemake_driver.sh config/pipeline_check.yaml              (all 5 stages)
#   sbatch hpc/scripts/snakemake_driver.sh config/pipeline_check.yaml train        (1 stage)
# Arguments after the config file go to snakemake: a stage name first, options after it.
# --set-resources on the command line replaces the whole set-resources block of hpc/profile/, so a
# full run must repeat every entry of that block next to its own values. The first full run was
# launched with these 13 entries:
#   sbatch -t 7-00:00:00 hpc/scripts/snakemake_driver.sh config/const_split_bott__seq256_knn7w_rank3.yaml \
#       --keep-going --set-resources train_model:slurm_partition=gpu train_model:gres=gpu:1 \
#       'train_model:constraint=l40s|a6000|a40' train_model:cpus_per_task=8 train_model:mem_mb=131072 \
#       train_model:runtime=1440 singer:runtime=180 diagnostics:mem_mb=32768 diagnostics:runtime=1440 \
#       diagnostics:cpus_per_task=8 export_trees:mem_mb=32768 export_trees:runtime=720 \
#       export_trees:cpus_per_task=8
# Snakemake's progress log goes to the driver's .err, not its .out.
# The driver only waits on the jobs, so 1 core and 4 GB are enough.
# Slurm does not create hpc/logs/: it must exist before sbatch.
# =============================================================================
#SBATCH -J argGNN_snakemake
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4G
#SBATCH -t 24:00:00
#SBATCH -o hpc/logs/%x_%j.out
#SBATCH -e hpc/logs/%x_%j.err

CONFIG=${1:?usage: sbatch hpc/scripts/snakemake_driver.sh CONFIG.yaml [STAGE] [snakemake options]}
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the repo root}" || exit 1
if [ ! -f Snakefile ] || [ ! -d src ]; then
    echo "!! $PWD is not the repo root (no Snakefile or no src/): submit from the repo root"
    exit 1
fi

module purge
unset LD_LIBRARY_PATH
module load python/3.13.5-dz6t || exit 1
source .venv/bin/activate
echo "python: $(python --version) at $(command -v python) | snakemake $(snakemake --version) | node $(hostname)"
echo "config: $CONFIG | extra arguments: ${*:2}"

# --verbose: logs each sbatch call, so the GPU request shows in this log
snakemake --profile hpc/profile --configfile "$CONFIG" --verbose "${@:2}"
