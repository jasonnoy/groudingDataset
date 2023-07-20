import sys
import os

import webdataset

print("adding", os.getcwd())
print("adding", os.path.join(os.getcwd(), "GLIP"))
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "GLIP"))


import argparse
import torch
import json
import spacy
from transformers import AutoTokenizer
from GLIP import *
from dataset_builder import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--master_addr', type=str, default='')
    parser.add_argument('--master_port', type=int, default=7878)
    args = parser.parse_args()

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    config_file = "/zhangpai21/jjh/projects/groundingDataset/GLIP/configs/pretrain/glip_Swin_L.yaml"
    weight_file = "/zhangpai21/checkpoints/GLIP-L/glip_large_model.pth"
    cfg.local_rank = args.local_rank
    cfg.num_gpus = 1
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda:{}".format(args.local_rank)])
    torch.cuda.set_device(args.local_rank)
    print("model device:",  cfg.MODEL.DEVICE)
    print("cuda device:", torch.cuda.current_device())
    rank = args.rank
    world_size = args.world_size

    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")

    glip_demo = GLIPDemo(
        cfg,
        confidence_threshold=0.7,
        show_mask_heatmaps=False,
        min_image_size=800,
        nlp = nlp
    )
    input_path = "/zhangpai21/webdataset/laion-aes/train"
    output_path = "/zhangpai21/webdataset/laion-aes/train/meta"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    total_files = get_id_list(input_path)
    part_size = len(total_files) // max((world_size-1), 1)
    start = rank*part_size
    end = min((rank+1)*part_size, len(total_files))
    files = total_files[start:end]
    # tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.LOCAL_PATH)
    for filename in files:
        tar_path = os.path.join(input_path, filename)
        res = {}
        batch_size = 7
        # laion_dataset = Laion(filename, input_path, nlp, tokenizer, transforms=glip_demo.transforms)
        laion_dataset = webdataset.WebDataset(os.path.join(input_path, filename))
        print("processing {}".format(filename))

        groundings = batch_parse_and_grounding_multi_class(glip_demo, laion_dataset, batch_size=batch_size, save_img=False, output_path=output_path)

        meta_filename = "{}.meta.jsonl".format(filename)
        output_meta_path = os.path.join(output_path, meta_filename)
        if os.path.exists(output_meta_path):
            os.remove(output_meta_path)
        with open(output_meta_path, 'a', encoding='utf-8') as f2:
            for i, grounding in enumerate(groundings):
                f2.write(json.dumps(grounding, ensure_ascii=False) + '\n')
        f2.close()
    print("done")
