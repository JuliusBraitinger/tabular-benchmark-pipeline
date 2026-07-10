#!/bin/bash
#SBATCH --job-name=pipeline
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=9G            
#SBATCH --partition=week            
#SBATCH --time=7-00:00:00            
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julius.braitinger@uni-tuebingen.de

set -euo pipefail

SCRATCH=/scratch/$SLURM_JOB_ID
mkdir -p "$SCRATCH/data"
export PIPELINE_CACHE="$SCRATCH"        # throwaway download caches (geoparse/kaggle/rnaseq)
export PIPELINE_DATA="$SCRATCH/data"    # datasets, tcga/uci/openml caches, rule_stats/soft_stats/html

export GEO_RNASEQ_DIR="$HOME/masterthesis/rnaseq_output"   # RNA-seq tar.gz exports from the nf pipeline
export CMD_DIR="$HOME/masterthesis/cmd_data"               # curatedMetagenomicData parquet (profiles + metadata)
export TABPFN_MODEL_CACHE_DIR="$HOME/masterthesis/tabpfn_cache"  # pre-warmed v2 weights on /home
export ENTREZ_EMAIL="julius.braitinger@uni-tuebingen.de"   # NCBI requires an email for Entrez (geo_array)
# export NCBI_API_KEY="your-key-here"   # optional: uncomment with a real key to lift rate limit 3->10 req/s
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate pipeline

cd "$SLURM_SUBMIT_DIR"   

DEST="$HOME/results/$SLURM_JOB_ID"
mkdir -p "$DEST"
# mirror scratch -> home every 30 min, so a crash/timeout doesnt kill everything 
( while true;
    do 
    rsync -a --exclude='cache/' "$PIPELINE_DATA"/ "$DEST/" || true; sleep 1800; done ) &

SYNC_PID=$!

python -m pipeline

kill "$SYNC_PID" 2>/dev/null || true
rsync -a --exclude='cache/' "$PIPELINE_DATA"/ "$DEST/" # final sync after pipeline finishes
echo "results copied to $DEST"
