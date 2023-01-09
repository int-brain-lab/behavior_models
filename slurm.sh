#!/bin/bash
#SBATCH --job-name=cv_behavior
#SBATCH --output=logs/slurm/cv_behavior.%A.%a.out
#SBATCH --error=logs/slurm/cv_behavior.%A.%a.err
#SBATCH --partition=public-cpu
#SBATCH --array=1-200
#SBATCH --mem=7000
#SBATCH --time=4-00:00:00

# source /home/users/f/findling/.bash_profile
# mamba activate iblenv

# extracting settings from $SLURM_ARRAY_TASK_ID
echo index $SLURM_ARRAY_TASK_ID

export PYTHONPATH="$PWD":$PYTHONPATH
# calling script

echo
python cross_validation_slurm.py $SLURM_ARRAY_TASK_ID
