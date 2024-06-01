#!/bin/bash -l
#
#SBATCH --job-name=uNetTwentyFiveEpochsBCE
#SBATCH --nodes=1
#SBATCH --time=19:30:00
#SBATCH --gres=gpu:1


module unload python
module load python/3.10-anaconda

export OUTDATED_IGNORE=1
export PATH="/home/hpc/iwso/iwso116h/.local/bin:$PATH"
module add tensorrt/8.6.1.6-cuda12.0-cudnn8.9
rsync -r $WORK/Data $TMPDIR


cd "$HOME"/emrad-analysis || exit
poetry run python main.py