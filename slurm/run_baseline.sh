#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=sqformer_baseline
#SBATCH --time=00:01:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH --gres=gpu:8

#SBATCH -o logs/baseline.%j.out
#SBATCH -e logs/baseline.%j.err

cd "${SLURM_SUBMIT_DIR}"

module purge || { echo "Failed to purge modules. Exiting."; exit 1; }
module load anaconda3      || { echo "Failed to load anaconda3. Exiting."; exit 1; }
module load nvidia/cuda/12 || { echo "Failed to load CUDA 12. Exiting."; exit 1; }

export VIRTUAL_ENV=/valhalla/projects/${SLURM_JOB_ACCOUNT}/conda_envs/torch
[ -d "${VIRTUAL_ENV}" ] || { echo "Missing venv: ${VIRTUAL_ENV}. Exiting."; exit 1; }
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

echo "Python: $(which python)"
echo "Accelerate: $(which accelerate)"
echo "Using ${SLURM_GPUS_ON_NODE} GPUs on this node."
/valhalla/projects/bg-eng-01/conda_envs/torch/bin/python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count())"
/valhalla/projects/bg-eng-01/conda_envs/torch/bin/python -c "import accelerate; print('Accelerate:', accelerate.__version__)"

cd /valhalla/projects/${SLURM_JOB_ACCOUNT}/LoreaEnc || { echo "Project folder not found. Exiting."; exit 1; }

mkdir -p logs outputs/baseline/1h outputs/baseline/100h outputs/baseline/960h

export MASTER_ADDR=localhost
export MASTER_PORT=12355
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export NUM_PROCESSES=${SLURM_GPUS_ON_NODE}

echo "Starting env-check run: 960h $(date)"
/valhalla/projects/bg-eng-01/conda_envs/torch/bin/accelerate launch   --num_processes ${NUM_PROCESSES}   --main_process_ip ${MASTER_ADDR}   --main_process_port ${MASTER_PORT}   SqueezeFormer/train.py     --hours 960     --seed 42     --output-dir outputs/baseline/960h     --run-name baseline_960h     --workers 8
[ $? -ne 0 ] && { echo "ERROR: env-check run failed."; exit 1; }

echo "Env check passed $(date)"
