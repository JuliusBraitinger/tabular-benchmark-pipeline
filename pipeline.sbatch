#!/bin/bash
#SBATCH --job-name=pipeline
#SBATCH --cpus-per-task=16          # max 24/node; A3's RandomForest uses n_jobs=-1
#SBATCH --mem-per-cpu=8G            # ~128 GB total — TCGA methylation matrices are huge
#SBATCH --partition=week           # day=24h; use week/month if including geo_array
#SBATCH --gres=gpu:1               # for TabPFN; drop --gres to schedule faster (mostly CPU job)
#SBATCH --time=72:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julius.braitinger@uni-tuebingen.de

set -euo pipefail

SCRATCH=/scratch/$SLURM_JOB_ID
mkdir -p "$SCRATCH/data"
export PIPELINE_CACHE="$SCRATCH"        # throwaway download caches (geoparse/kaggle/rnaseq)
export PIPELINE_DATA="$SCRATCH/data"    # datasets, tcga/uci/openml caches, rule_stats/soft_stats/html

export GEO_RNASEQ_DIR="$HOME/data/rnaseq"   # not in .env; only if you uploaded RNA-seq exports
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tcml

cd "$SLURM_SUBMIT_DIR"                  # run from the cloned repo
python -m pipeline

DEST="$HOME/results/$SLURM_JOB_ID"
mkdir -p "$DEST"
cp -R "$PIPELINE_DATA"/. "$DEST/"       # datasets + all csv/html, one shot
echo "results copied to $DEST"
