#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=6G
#SBATCH --partition=day
#SBATCH --time=1-00:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julius.braitinger@uni-tuebingen.de

set -euo pipefail

export BENCH_DATA="$HOME/results/2203556/datasets"   # the run whose datasets we benchmark
# cpu, not cuda: the pipeline run 2203556 proved TabPFN works on cpu here, while the
# benchmark scored it 0/25 and 0/15 on cuda. no gpu request also means any node will do.
export BENCH_DEVICE=cpu
export TABPFN_MODEL_CACHE_DIR="$HOME/masterthesis/tabpfn_cache"  # pre-warmed weights, no download

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate pipeline

cd "$SLURM_SUBMIT_DIR"

# fail here with a clear message instead of a cryptic one 20 minutes in.
# none of these are in environment.yml, install them once on the login node:
#   conda activate pipeline && pip install nbconvert ipykernel lightgbm xgboost tabpfnwide
for mod in nbconvert ipykernel lightgbm xgboost tabpfn tabpfnwide; do
    python -c "import $mod" 2>/dev/null || { echo "MISSING: $mod - see comment above"; exit 1; }
done

DEST="$HOME/results/benchmark_$SLURM_JOB_ID"
mkdir -p "$DEST"

# the notebook appends to elo_scores.csv after every dataset, so copy it out every
# 10 min - a timeout then still leaves the datasets that already finished
( while true; do cp -f elo_scores.csv "$DEST/" 2>/dev/null || true; sleep 600; done ) &
SYNC_PID=$!

# copy whatever exists even if the notebook crashes half way through
trap 'kill "$SYNC_PID" 2>/dev/null || true; cp -f elo_scores.csv "$DEST/" 2>/dev/null || true' EXIT

jupyter nbconvert --to notebook --execute benchmark_elo.ipynb \
    --output "benchmark_elo_$SLURM_JOB_ID.ipynb" \
    --ExecutePreprocessor.timeout=-1

cp -f "benchmark_elo_$SLURM_JOB_ID.ipynb" "$DEST/"
echo "results copied to $DEST"
