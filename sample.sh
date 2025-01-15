#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1  # Request 1 A100 GPU
#SBATCH -c 48              # Request 4 CPU cores
#SBATCH --mem=500GB        # Request 16GB memory
#SBATCH -t 10-20:30:00     # Set a 30-minute time limit
#SBATCH -J EpsOptimizd     # Job name
#SBATCH -o Slurm_Outputs/EpsilonV_Opt.out   # Output file for logs

# Activate the SHAP virtual environment
source SHAP/bin/activate

# Run the Python scripts in parallel (hard-coded paths)
python3 Sbatch/EpsConvCif/t2e3_ConvCif.py &
python3 Sbatch/EpsConvCif/t2e4_ConvCif.py &
python3 Sbatch/EpsConvCif/t2e5_ConvCif.py &
python3 Sbatch/EpsConvCif/t2e6_ConvCif.py &
python3 Sbatch/EpsConvCif/t2e7_ConvCif.py &
python3 Sbatch/EpsConvCif/t2e8_ConvCif.py &
python3 Sbatch/EpsResCif/t2e3_ResCif.py &
python3 Sbatch/EpsResCif/t2e4_ResCif.py &
python3 Sbatch/EpsResCif/t2e5_ResCif.py &
python3 Sbatch/EpsResCif/t2e6_ResCif.py &
python3 Sbatch/EpsResCif/t2e7_ResCif.py &
python3 Sbatch/EpsResCif/t2e8_ResCif.py &




# Wait for all scripts to finish
wait

# Deactivate the virtual environment
deactivate
