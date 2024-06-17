#!/bin/bash -l
#
#SBATCH --job-name=uNetRadarcadiaImage50WaveletTypes
#SBATCH --nodes=1
#SBATCH --time=8:30:00
#SBATCH --gres=gpu:1


module unload python
module load python/3.10-anaconda

export OUTDATED_IGNORE=1
export PATH="/home/hpc/iwso/iwso116h/.local/bin:$PATH"
module add tensorrt/8.6.1.6-cuda12.0-cudnn8.9
rsync -r $HPCVAULT/DataRadarcadia $TMPDIR/Data

cd "$HOME"/emrad-analysis || exit
poetry run python main.py --epochs 50 --learning_rate 0.0001 --image_based False --datasource radarcadia --breathing_type all --label_type ecg --dual_channel False --log False --wavelet gaus1
poetry run python main.py --epochs 50 --learning_rate 0.0001 --image_based False --datasource radarcadia --breathing_type all --label_type gausian --dual_channel False --log True --wavelet gaus1
poetry run python main.py --epochs 50 --learning_rate 0.0001 --image_based False --datasource radarcadia --breathing_type all --label_type ecg --dual_channel True --wavelet gaus1

poetry run python main.py --epochs 50 --learning_rate 0.0001 --image_based False --datasource radarcadia --breathing_type all --label_type ecg --dual_channel False --log False --wavelet mexh
poetry run python main.py --epochs 50 --learning_rate 0.0001 --image_based False --datasource radarcadia --breathing_type all --label_type gausian --dual_channel False --log True --wavelet mexh
poetry run python main.py --epochs 50 --learning_rate 0.0001 --image_based False --datasource radarcadia --breathing_type all --label_type ecg --dual_channel True --wavelet mexh

poetry run python main.py --epochs 50 --learning_rate 0.0001 --image_based False --datasource radarcadia --breathing_type all --label_type ecg --dual_channel False --log False --wavelet shan1-1
poetry run python main.py --epochs 50 --learning_rate 0.0001 --image_based False --datasource radarcadia --breathing_type all --label_type gausian --dual_channel False --log True --wavelet shan1-1
poetry run python main.py --epochs 50 --learning_rate 0.0001 --image_based False --datasource radarcadia --breathing_type all --label_type ecg --dual_channel True --wavelet shan1-1