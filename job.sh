#!/bin/bash -l
#
#SBATCH --job-name=uNet75BCEWavelet0001Sigmoid
#SBATCH --nodes=1
#SBATCH --time=22:45:00
#SBATCH --gres=gpu:rtx3080:1


module unload python
module load python/3.10-anaconda

export OUTDATED_IGNORE=1
export PATH="/home/hpc/iwso/iwso116h/.local/bin:$PATH"
module add tensorrt/8.6.1.6-cuda12.0-cudnn8.9
rsync -r $WORK/Data $TMPDIR


cd "$HOME"/altPreprocessing/emrad-analysis || exit
poetry run python main.py