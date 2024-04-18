#!/bin/bash -l
#
#SBATCH --job-name=secondTry
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

cd "$HOME"/emrad-analysis || exit

module unload python
module module load python/3.10-anaconda

export "XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_ROOT"

module unload cuda
module add cuda/11.8.0

module add tensorrt/8.6.1.6-cuda11.8-cudnn8.9

export OUTDATED_IGNORE=1
export DATA_PATH="/home/vault/empkins/tpD/D03/Data/MA_Simon_Meske/Data_D02/data_per_subject"

poetry run python main.py