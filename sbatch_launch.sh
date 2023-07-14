#!/bin/bash
#SBATCH --job-name=slurm_example
#SBATCH --output=slurm_example_%j.out
#SBATCH --error=slurm_example_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=dev
#SBATCH --export=ALL

srun single_launch.sh 
echo "Done with job $SLURM_JOB_ID"

