#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=01:00:00

module load 2024
module load CUDA/12.6.0
module load Ninja/1.12.1-GCCcore-13.3.0

uv run python pytorch-nd-semiconv/examples/ops_equivalent.py