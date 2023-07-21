#!/bin/bash
#SBATCH --job-name=build_bounding_dataset_laion_115m
#SBATCH --output=./logs/build_bounding_dataset_laion_115m_%j.out
#SBATCH --error=./logs/build_bounding_dataset_laion_115m_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --partition=dev
#SBATCH --export=ALL

srun experiments/build_from_laion_115m_ningxia.sh
#srun experiments/build_from_laion_aes_jinan.sh
echo "Done with job $SLURM_JOB_ID"
