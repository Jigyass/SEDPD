#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1                   # Request 1 A100 GPU
#SBATCH -c 32                               # Request 4 CPU cores
#SBATCH --mem=100GB                         # Request 16GB memory
#SBATCH -t 10-20:30:00                      # Set a 30-minute time limit
#SBATCH -J DefLimeExp                       # Job name
#SBATCH -o Slurm_Outputs/ConLimeCifDefExp.out  # Output file for logs

# Activate the SHAP virtual environment
source SHAP/bin/activate

# Run the Python scripts in parallel (hard-coded paths)
python3 Explan_Values_Mod/LimeCifE2.py
python3 Explan_Values_Mod/LimeCifE3.py
python3 Explan_Values_Mod/LimeCifE4.py
python3 Explan_Values_Mod/LimeCifE5.py
python3 Explan_Values_Mod/LimeCifE6.py
python3 Explan_Values_Mod/LimeCifE7.py
python3 Explan_Values_Mod/LimeCifE8.py



# Wait for all scripts to finish
wait

# Deactivate the virtual environment
deactivate
