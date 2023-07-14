import os

import torch
import argparse

if __name__ == '__main__':
    # read distributed training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--master_addr', type=str, default='')
    parser.add_argument('--master_port', type=int, default=7878)

    args = parser.parse_args()
    print(args)
    # set torch distributed
    tensor = torch.tensor([args.rank]).cuda(device=args.local_rank)
    print('rank: {}, tensor device: {}'.format(args.rank, tensor.device))
