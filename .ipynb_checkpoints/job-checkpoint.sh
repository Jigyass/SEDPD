#!/bin/bash

#SBATCH -p gpu                    # Specify the GPU partition
#SBATCH --gres=gpu:a100:1         # Request 1 A100 GPU
#SBATCH -c 16                      # Request 4 CPU cores
#SBATCH --mem=80GB                # Request 8GB memory
#SBATCH -t 5-23:20:00             # 5 hour time limit
#SBATCH -J Defense                # Name of the job
#SBATCH -o slurm-%j.out           # Save output to slurm-<job_id>.out

# Load modules (if needed) - Uncomment and customize as required
# module load cuda/11.3

# Activate virtual environment
source SHAP/bin/activate

# Print details about the job
echo "Job ${SLURM_JOB_ID} running on ${HOSTNAME}"

# Run your Python script
python3 Defense.py

# Optional: Deactivate virtual environment (cleanup)
deactivate
