#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1  # Request 1 A100 GPU
#SBATCH -c 16              # Request 4 CPU cores
#SBATCH --mem=128GB        # Request 16GB memory
#SBATCH -t 5-20:30:00      # Set a 30-minute time limit
#SBATCH -J SHAP_Sample     # Job name
#SBATCH -o LimeSm-%j.out   # Output file for logs

# Activate the SHAP virtual environment
source SHAP/bin/activate

# Run the Python scripts in parallel (hard-coded paths)
python3 SamplingDicts_Lime/t2e1.py &
python3 SamplingDicts_Lime/t2e2.py &
python3 SamplingDicts_Lime/t2e3.py &
python3 SamplingDicts_Lime/t2e4.py &
python3 SamplingDicts_Lime/t2e5.py &
python3 SamplingDicts_Lime/t2e6.py &
python3 SamplingDicts_Lime/t2e7.py &
python3 SamplingDicts_Lime/t2e8.py &

# Wait for all scripts to finish
wait

# Deactivate the virtual environment
deactivate
