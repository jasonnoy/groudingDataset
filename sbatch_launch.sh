#!/bin/bash
#SBATCH --job-name=CoM_grounding
#SBATCH --output=./logs/CoM_grounding_%j.out
#SBATCH --error=./logs/CoM_grounding_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --export=ALL

srun experiments/build_from_CoM_zhongwei.sh
#srun experiments/build_from_laion_115m_ningxia.sh
#srun experiments/build_from_laion_aes_jinan.sh
echo "Done with job $SLURM_JOB_ID"
