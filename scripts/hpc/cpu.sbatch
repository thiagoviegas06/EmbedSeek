#!/bin/bash
# Run one Python entry point on a CPU node, inside the same image + overlay as embed.sbatch (see its header).
# Submit from the repo root, on a login node:
#   mkdir -p logs && sbatch scripts/hpc/cpu.sbatch scripts/encoder_diagnostics.py data/jigsaw
# Override resources on the command line, e.g. sbatch --time=04:00:00 --cpus-per-task=16 scripts/hpc/cpu.sbatch ...
#SBATCH --job-name=embedseek-cpu
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
[ $# -ge 1 ] || { echo "usage: sbatch scripts/hpc/cpu.sbatch <script.py> [args...]"; exit 2; }

SING=${SING:-/share/apps/apptainer/bin/singularity}
IMG=${IMG:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}
OVL=${OVL:-/scratch/$USER/neuro3.ext3}

cd "$SLURM_SUBMIT_DIR"
echo "cmd=$* host=$(hostname) cpus=$SLURM_CPUS_PER_TASK start=$(date)"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

"$SING" exec --overlay "$OVL:ro" "$IMG" /bin/bash -c '
  set -euo pipefail
  source .venv/bin/activate
  export HF_HOME=${SCRATCH:-$HOME}/hf_cache   # model downloads go to scratch, not the home quota
  python -I "$@"
' bash "$@"
echo "done=$(date)"
