#!/bin/bash -l
#
#SBATCH --job-name=radarCadiaFirstHalf
#SBATCH --nodes=1
#SBATCH --time=19:30:00
#SBATCH --gres=gpu:1


module unload python
module load python/3.10-anaconda

export OUTDATED_IGNORE=1
export PATH="/home/hpc/iwso/iwso116h/.local/bin:$PATH"
module add tensorrt/8.6.1.6-cuda12.0-cudnn8.9
rsync -r $HPCVAULT/DataRadarcadia $TMPDIR/Data

cd "$HOME"/emrad-analysis || exit
poetry run python main.py --epochs 50 --learning_rate 0.0001 --image_based False --datasource radarcadia --breathing_type all --label_type gaussian --dual_channel False --log False --loss bce --wavelet morl
poetry run python main.py --epochs 50 --learning_rate 0.001 --image_based False --datasource radarcadia --breathing_type all --label_type gaussian --dual_channel False --log False --loss bce --wavelet morl
poetry run python main.py --epochs 50 --learning_rate 0.0001 --image_based False --datasource radarcadia --breathing_type all --label_type gaussian --dual_channel True --loss bce --wavelet morl
poetry run python main.py --epochs 50 --learning_rate 0.001 --image_based False --datasource radarcadia --breathing_type all --label_type gaussian --dual_channel True --loss bce --wavelet morl

poetry run python main.py --epochs 50 --learning_rate 0.0001 --image_based False --datasource radarcadia --breathing_type all --label_type ecg --dual_channel False --log False --loss mse --wavelet morl
poetry run python main.py --epochs 50 --learning_rate 0.001 --image_based False --datasource radarcadia --breathing_type all --label_type ecg --dual_channel False --log False --loss mse --wavelet morl
poetry run python main.py --epochs 50 --learning_rate 0.001 --image_based False --datasource radarcadia --breathing_type all --label_type ecg --dual_channel True --loss mse --wavelet morl




poetry run python main.py --epochs 50 --learning_rate 0.0001 --image_based False --datasource radarcadia --breathing_type all --label_type gaussian --dual_channel False --log False --loss bce --wavelet gaus1
poetry run python main.py --epochs 50 --learning_rate 0.001 --image_based False --datasource radarcadia --breathing_type all --label_type gaussian --dual_channel False --log False --loss bce --wavelet gaus1
poetry run python main.py --epochs 50 --learning_rate 0.0001 --image_based False --datasource radarcadia --breathing_type all --label_type gaussian --dual_channel True --loss bce --wavelet gaus1
poetry run python main.py --epochs 50 --learning_rate 0.001 --image_based False --datasource radarcadia --breathing_type all --label_type gaussian --dual_channel True --loss bce --wavelet gaus1

poetry run python main.py --epochs 50 --learning_rate 0.0001 --image_based False --datasource radarcadia --breathing_type all --label_type ecg --dual_channel False --log False --loss mse --wavelet gaus1
poetry run python main.py --epochs 50 --learning_rate 0.001 --image_based False --datasource radarcadia --breathing_type all --label_type ecg --dual_channel False --log False --loss mse --wavelet gaus1
poetry run python main.py --epochs 50 --learning_rate 0.001 --image_based False --datasource radarcadia --breathing_type all --label_type ecg --dual_channel True --loss mse --wavelet gaus1


poetry run python main.py --epochs 50 --learning_rate 0.0001 --image_based False --datasource radarcadia --breathing_type all --label_type gaussian --dual_channel False --log False --loss bce --wavelet mexh
poetry run python main.py --epochs 50 --learning_rate 0.001 --image_based False --datasource radarcadia --breathing_type all --label_type gaussian --dual_channel False --log False --loss bce --wavelet mexh
poetry run python main.py --epochs 50 --learning_rate 0.0001 --image_based False --datasource radarcadia --breathing_type all --label_type gaussian --dual_channel True --loss bce --wavelet mexh
poetry run python main.py --epochs 50 --learning_rate 0.001 --image_based False --datasource radarcadia --breathing_type all --label_type gaussian --dual_channel True --loss bce --wavelet mexh