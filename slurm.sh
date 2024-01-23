#!/bin/bash
#SBATCH --job-name=behavior
#SBATCH --output=logs/slurm/behavior.%A.%a.out
#SBATCH --error=logs/slurm/behavior.%A.%a.err
#SBATCH --partition=public-cpu
#SBATCH --array=1-140
#SBATCH --mem=7000
#SBATCH --time=96:00:00
# 1300/4=325

# source /home/users/f/findling/.bash_profile
# mamba activate iblenv

# extracting settings from $SLURM_ARRAY_TASK_ID
echo index $SLURM_ARRAY_TASK_ID

export PYTHONPATH="$PWD":$PYTHONPATH
# calling script

echo
python run_cv_fit_slurm.py $SLURM_ARRAY_TASK_ID
