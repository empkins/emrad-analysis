#!/bin/bash -l
#
#SBATCH --job-name=uNetCombinedGaus1_40Epochs0001
#SBATCH --nodes=1
#SBATCH --time=10:30:00
#SBATCH --gres=gpu:rtx3080:1


module unload python
module load python/3.10-anaconda

export OUTDATED_IGNORE=1
export PATH="/home/hpc/iwso/iwso116h/.local/bin:$PATH"
module add tensorrt/8.6.1.6-cuda12.0-cudnn8.9
tar -xf $WORK/CombinedData.tar -C $TMPDIR/

cd "$HOME"/altPreprocessing/emrad-analysis || exit
poetry run python main.py --epochs 40 --learning_rate 0.0001 --image_based False --log False --label_type gaussian --dual_channel False --wavelet gaus1 --combined True





