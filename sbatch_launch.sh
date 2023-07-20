#!/bin/bash
#SBATCH --job-name=build_bounding_dataset_laion_aes
#SBATCH --output=slurm_example_%j.out
#SBATCH --error=slurm_example_%j.err
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --partition=dev
#SBATCH --export=ALL
#SBATCH --exclude=1hlo8k81jmmul-0,1m5qco0g209qm-0

srun experiments/build_from_laion_aes_jinan.sh
echo "Done with job $SLURM_JOB_ID"
