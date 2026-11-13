#!/bin/bash
# Embed every split with each encoder on one GPU; one array task per model.
# Submit from the repo root, on a login node (sbatch is not available inside the singularity container):
#   mkdir -p logs && sbatch scripts/hpc/embed.sbatch
# Defaults are for NYU Torch: .venv was built from the overlay's /ext3/miniforge3 python, so the job runs it
# inside the same image + overlay. The overlay is mounted read-only, so it must not be open read-write by an
# interactive session while the job runs. Override with OVL=/path/to/other.ext3 sbatch ... (and --account=...).
#SBATCH --job-name=embedseek-embed
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=a100_tandon
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

SING=${SING:-/share/apps/apptainer/bin/singularity}
IMG=${IMG:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}
OVL=${OVL:-/scratch/$USER/neuro3.ext3}

cd "$SLURM_SUBMIT_DIR"
echo "task=$SLURM_ARRAY_TASK_ID model=$MODEL host=$(hostname) start=$(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

"$SING" exec --nv --overlay "$OVL:ro" "$IMG" /bin/bash -c "
  set -euo pipefail
  source .venv/bin/activate
  export HF_HOME=\${SCRATCH:-\$HOME}/hf_cache   # model downloads go to scratch, not the home quota
  python -I scripts/embed.py data/jigsaw/splits data/jigsaw/emb '$MODEL' 128
"
echo "done=$(date)"
