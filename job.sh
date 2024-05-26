#!/bin/bash -l
#
#SBATCH --job-name=uNetFiftyEpochs
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1

cd "$HOME"/emrad-analysis || exit

module unload python
module load python/3.10-anaconda

export OUTDATED_IGNORE=1
export PATH="/home/hpc/iwso/iwso116h/.local/bin:$PATH"
module add tensorrt/8.6.1.6-cuda12.0-cudnn8.9
rsync -r $WORK/Data $TMPDIR

poetry run python main.py