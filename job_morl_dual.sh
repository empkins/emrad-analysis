#!/bin/bash -l
#
#SBATCH --job-name=uNetD02MorlDual
#SBATCH --nodes=1
#SBATCH --time=21:30:00
#SBATCH --gres=gpu:rtx3080:1


module unload python
module load python/3.10-anaconda

export OUTDATED_IGNORE=1
export PATH="/home/hpc/iwso/iwso116h/.local/bin:$PATH"
module add tensorrt/8.6.1.6-cuda12.0-cudnn8.9
rsync -r $WORK/DataD02 $TMPDIR

cd "$HOME"/emrad-analysis || exit

poetry run python main.py --epochs 40 --learning_rate 0.0001 --image_based False --datasource d02 --label_type gaussian --dual_channel True --wavelet morl
poetry run python main.py --epochs 40 --learning_rate 0.001 --image_based False --datasource d02 --label_type gaussian --dual_channel True --wavelet morl
