#!/bin/bash
# Embed every split with each encoder on one GPU; one array task per model.
# Submit from the repo root:  mkdir -p logs && sbatch scripts/hpc/embed.sbatch
# Fill in the account (and partition, if your cluster needs one) before submitting.
#SBATCH --job-name=embedseek-embed
#SBATCH --account=CHANGE_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

MODELS=(
  sentence-transformers/all-mpnet-base-v2
  BAAI/bge-base-en-v1.5
  sentence-transformers/all-MiniLM-L6-v2   # re-embedded on the same GPU: comparable C_embed timings
)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export HF_HOME=${SCRATCH:-$HOME}/hf_cache   # model downloads go to scratch, not the home quota

nvidia-smi --query-gpu=name,memory.total --format=csv
python -I scripts/embed.py data/jigsaw/splits data/jigsaw/emb "$MODEL" 128
