#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:T4:1
#SBATCH --time=24:00:00
#SBATCH --mem=256GB
#SBATCH --nodes=2
#SBATCH --job-name=pytest

# Set the path to the virtual environment
# export VENV_PATH="/work/data_science/ODU_CAPSTONE_2025/sns_venv/"

# Activate the virtual environment
source /work/data_science/ODU_CAPSTONE_2025/sns_env/bin/activate

# Run Python commands within the virtual environment
python ~/SNS_Anomaly_Detection/cnn_lstm/driver.py --train
python ~/SNS_Anomaly_Detection/cnn_lstm/driver.py --test
