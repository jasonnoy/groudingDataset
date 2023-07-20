#!/bin/bash
#SBATCH --job-name=build_bounding_dataset_laion_aes
#SBATCH --output=./logs/build_bounding_dataset_laion_aes_%j.out
#SBATCH --error=./logs/build_bounding_dataset_laion_aes_%j.err
#SBATCH --nodes=8
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --partition=dev
#SBATCH --export=ALL
#SBATCH --exclude=1hlo8k81jmmul-0,1m5qco0g209qm-0,2fpm7g8mhc7m7-0,c89rgca402ma5-0

srun experiments/build_from_laion_aes_jinan.sh
echo "Done with job $SLURM_JOB_ID"
