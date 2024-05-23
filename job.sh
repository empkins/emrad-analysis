#!/bin/bash -l
#
#SBATCH --job-name=uNetConvFiveEpochs
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1

cd "$HOME"/emrad-analysis || exit

module unload python
module load python/3.10-anaconda

export OUTDATED_IGNORE=1
export PATH="/home/hpc/iwso/iwso116h/.local/bin:$PATH"

poetry run python main.py