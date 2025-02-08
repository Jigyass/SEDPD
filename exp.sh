#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1                   # Request 1 A100 GPU
#SBATCH -c 92                               # Request 4 CPU cores
#SBATCH --mem=500GB                         # Request 16GB memory
#SBATCH -t 10-20:30:00                      # Set a 30-minute time limit
#SBATCH -J DefResExp                           # Job name
#SBATCH -o Slurm_Outputs/ResShapMniDefExp.out  # Output file for logs

# Activate the SHAP virtual environment
source SHAP/bin/activate

# Run the Python scripts in parallel (hard-coded paths)
python3 Explan_Values_Mod/ShapMniE2.py
python3 Explan_Values_Mod/ShapMniE3.py
python3 Explan_Values_Mod/ShapMniE4.py
python3 Explan_Values_Mod/ShapMniE5.py
python3 Explan_Values_Mod/ShapMniE6.py
python3 Explan_Values_Mod/ShapMniE7.py
python3 Explan_Values_Mod/ShapMniE8.py



# Wait for all scripts to finish
wait

# Deactivate the virtual environment
deactivate
