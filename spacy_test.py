import argparse
import torch
import json
import os
import re
import sys
import spacy
from tqdm import tqdm
import webdataset
import warnings
import math
warnings.filterwarnings("ignore")
from dataset_builder import *


def split_list_by_n(origin_list, n):
    step = math.ceil(len(origin_list) / n)
    res = []
    for i in range(0, len(origin_list), step):
        res.append(origin_list[i:i + step])
    return res


def get_entity_offset(caption, entities):
    offsets = []
    for entity in entities:
        # want no overlays
        found = {(0, 0)}
        try:
            for m in re.finditer(entity, caption):
                if (m.start(), m.end()) not in found:
                    found.add((m.start(), m.end()))
            offsets.append(len(found) - 2)
        except Exception as e:
            raise ValueError("caption:{}, entity:{}".format(caption, entity))
    return offsets


def get_entities(text):
    doc = nlp(text)
    return doc.noun_chunks


def get_all_entity_map(entity_lists):
    res = []
    for entity_list in entity_lists:
        res.extend(entity_list)
    # return res, dict(zip(res, range(len(res))))
    return res


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--master_addr', type=str, default='')
    parser.add_argument('--master_port', type=int, default=7878)
    args = parser.parse_args()

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    config_file = "GLIP/configs/pretrain/glip_Swin_L.yaml"
    weight_file = "/nxchinamobile2/shared/official_pretrains/hf_home/GLIP-L/glip_large_model.pth"
    torch.cuda.set_device(args.local_rank)
    rank = args.rank
    local_rank = args.local_rank
    world_size = args.world_size

    # spacy.prefer_gpu(args.local_rank)
    nlp = spacy.load("en_core_web_trf")
    path = "/nxchinamobile2/shared/jjh/laion115m_grounding/part-00032/3200467.meta.jsonl"
    out_path = "/nxchinamobile2/shared/jjh/laion115m-debug/part-00032/3200467.meta.jsonl"
    with open(path, "r", encoding='utf-8') as f, open(out_path, 'a', encoding='utf-8') as f2:
        count = 0
        captions = []
        datas = []
        for line in f:
            data = json.loads(line)
            if data['status'] == 'success':
                caption = data["caption"]
                captions.append(caption)
            datas.append(data)
            count += 1
            if count == 20:
                entity_lists = [get_entities(caption) for caption in captions]
                entity_offsets = [get_entity_offset(cap, entities) for cap, entities in zip(captions, entity_lists)]
                all_entities = get_all_entity_map(entity_lists)
                assert len(entity_lists) == len(entity_offsets)
                all_idx = 0
                for i in range(len(entity_lists)):
                    groundings = {}
                    for j in range(len(entity_lists[i])):
                        old_entity = entity_lists[i][j]
                        offset = entity_offsets[i][j]
                        new_entity = all_entities[all_idx - offset]
                        groundings[new_entity] = data[old_entity]
                        all_idx += 1
                    datas[i]['groundings'] = groundings
                    f2.write(json.dumps(datas[i]) + '\n')
                f2.close()
                f.close()
                count = 0
                captions = []
                datas = []
