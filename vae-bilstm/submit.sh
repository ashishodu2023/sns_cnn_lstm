#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:T4:1
#SBATCH --time=24:00:00
#SBATCH --mem=256GB
#SBATCH --nodes=1
#SBATCH --job-name=pytest

# Set the path to the virtual environment
# export VENV_PATH="/work/data_science/ODU_CAPSTONE_2025/sns_venv/"

# Activate the virtual environment
source /work/data_science/ODU_CAPSTONE_2025/sns_env/bin/activate

# Run Python commands within the virtual environment
python ~/SNS_Anomaly_Detection/vae_bilstm/driver.py train --epochs 10 --batch_size 8 --learning_rate 1e-4 --latent_dim 32 --model_path ~/ODU_CAPSTONE_2025/vae_bilstm/vae_bilstm_model.weights.h5 --tensorboard_logdir logs/fit
python driver.py predict --model_path ~/SNS_Anomaly_Detection/vae_bilstm/vae_bilstm_model.weights.h5 --threshold_percentile '99.9'
