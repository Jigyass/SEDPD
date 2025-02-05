#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1                   # Request 1 A100 GPU
#SBATCH -c 96                               # Request 4 CPU cores
#SBATCH --mem=500GB                         # Request 16GB memory
#SBATCH -t 10-20:30:00                      # Set a 30-minute time limit
#SBATCH -J DefenseData                      # Job name
#SBATCH -o Slurm_Outputs/ResCifDefData.out  # Output file for logs

# Activate the SHAP virtual environment
source SHAP/bin/activate

# Run the Python scripts in parallel (hard-coded paths)
python3 Sbatch/Mnist_ConvSHAP.py &
python3 Sbatch/DTE4.py &
python3 Sbatch/DTE5.py &
python3 Sbatch/DTE6.py &
python3 Sbatch/DTE7.py &
python3 Sbatch/DTE8.py &



# Wait for all scripts to finish
wait

# Deactivate the virtual environment
deactivate
