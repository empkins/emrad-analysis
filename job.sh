#!/bin/bash -l
#
#SBATCH --job-name=ModelTraining
#SBATCH --nodes=1
#SBATCH --time=7:00:00
#SBATCH --gres=gpu:1

cd "$HOME"/emrad-analysis || exit

module unload python
module load python/3.10-anaconda

export OUTDATED_IGNORE=1
export DATA_PATH="/home/woody/iwso/iwso116h/Data"
export PATH="/home/hpc/iwso/iwso116h/.local/bin:$PATH"

cp -r "$WORK/Data" "$TMPDIR"

poetry run python main.py