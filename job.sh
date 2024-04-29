#!/bin/bash -l
#
#SBATCH --job-name=ExtendedRun
#SBATCH --nodes=1
#SBATCH --time=11:00:00
#SBATCH --gres=gpu:1

cd "$HOME"/emrad-analysis || exit

module unload python
module load python/3.10-anaconda

export OUTDATED_IGNORE=1
export DATA_PATH="/home/vault/empkins/tpD/D03/Data/MA_Simon_Meske/Data_D02/data_per_subject"
export PATH="/home/hpc/iwso/iwso116h/.local/bin:$PATH"

poetry run python main.py