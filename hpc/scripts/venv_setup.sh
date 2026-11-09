#!/bin/bash
# =============================================================================
# venv_setup.sh: build the argGNN venv on Oscar.
# Run: bash hpc/scripts/venv_setup.sh
#
# Installs from requirements-frozen-oscar.txt when it exists.
# Otherwise it resolves requirements.txt and writes requirements-frozen-oscar.txt.
# =============================================================================
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)   # the repo root, 2 levels above this file
cd "$ROOT" || exit 1

module purge
unset LD_LIBRARY_PATH
module load python/3.13.5-dz6t || exit 1

rm -rf "${ROOT:?}/.venv"
python3 -m venv .venv || exit 1
source .venv/bin/activate
echo "venv python: $(python --version) at $(command -v python)"
python -m pip install --upgrade pip || exit 1

if [ -f requirements-frozen-oscar.txt ]; then
    echo "Installing from requirements-frozen-oscar.txt"
    echo "  (delete requirements-frozen-oscar.txt to force a fresh resolve from requirements.txt)"
    python -m pip install -r requirements-frozen-oscar.txt || exit 1
else
    echo "Installing from requirements.txt"
    python -m pip install -r requirements.txt || exit 1
    python -m pip freeze > requirements-frozen-oscar.txt || exit 1
    echo "wrote $ROOT/requirements-frozen-oscar.txt"
fi

echo "snakemake $(snakemake --version)"
