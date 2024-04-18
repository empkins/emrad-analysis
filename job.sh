#!/bin/bash -l
#
#SBATCH --job-name=firstRun
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

cd "$WORK"/emrad-analysis || exit

module unload python
module module load python/3.10-anaconda

poetry install
export "XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_ROOT"

module unload cuda
module add cuda/11.8.0

module add tensorrt/8.6.1.6-cuda11.8-cudnn8.9

export OUTDATED_IGNORE=1
DATA_PATH="/home/vault/empkins/tpD/D03/Data/MA_Simon_Meske/Data_D02"

poetry run python main.py