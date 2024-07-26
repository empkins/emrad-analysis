#!/bin/bash -l
#
#SBATCH --job-name=magnitude_d02_001
#SBATCH --nodes=1
#SBATCH --time=19:30:00
#SBATCH --gres=gpu:1

module unload python
module load python/3.10-anaconda

export OUTDATED_IGNORE=1
export PATH="/home/hpc/iwso/iwso116h/.local/bin:$PATH"
module add tensorrt/8.6.1.6-cuda12.0-cudnn8.9
rsync -r $HPCVAULT/DataD02 $TMPDIR/Data

cd "$HOME"/altPreprocessing/emrad-analysis || exit
poetry run python main.py --epochs 350 --learning_rate 0.001 --image_based False --datasource d02 --label_type gaussian --wavelet mag --mag True