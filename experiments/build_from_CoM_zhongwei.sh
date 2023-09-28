#!/bin/bash
# author: mingding

# this is launched by srun
# command for this script: srun -N 2 --gres=gpu:2 --ntasks-per-node=2 --cpus-per-task=4 --job-name=slurm_example --partition=dev --time=00:10:00 --output=slurm_example.out --error=slurm_example.err ./single_launch.sh

# if SLURM defined, set by SLURM environment
module unload cuda
module unload cuda/11.8
module load cuda/11.7

export CUDA_HOME=/share/apps/cuda-11.7

WORLD_SIZE=${SLURM_NTASKS:-1}
RANK=${SLURM_PROCID:-0}
# MASTER_ADDR is the first in SLURM_NODELIST
if [ -z "$SLURM_NODELIST" ]; then
    MASTER_ADDR=localhost
    MASTER_PORT=7878
else
    MASTER_ADDR=`scontrol show hostnames $SLURM_NODELIST | head -n 1`
    MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
fi
# generate a port at random
LOCAL_RANK=${SLURM_LOCALID:-0}

echo "RANK=$RANK RUN on `hostname`, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export HF_HOME=/shared/official_pretrains/hf_home
export SAT_HOME=/shared/official_pretrains/sat_home
export TRANSFORMERS_CACHE=/share/home/jijunhui/transformers_cache
export HF_MODULES_CACHE=/share/home/jijunhui/hf_modules_cache
# shellcheck disable=SC2046
python scripts/build_from_CoM.py --world_size $WORLD_SIZE --rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT --local_rank $LOCAL_RANK --batch_size 10 --thresh 0.4
# --save_img
echo "DONE for RANK=$RANK on `hostname`"