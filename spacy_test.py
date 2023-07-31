import argparse
import torch
import json
import os
import re
import spacy
from tqdm import tqdm
from multiprocessing import Process
import math


def remove_punctuation(text: str) -> str:
    punct = ['|', ':', ';', '@', '(', ')', '[', ']', '{', '}', '^', '\\', '/',
             '\'', '\"', '’', '`', '?', '$', '%', '#', '!', '&', '*', '+', ',', '.'
             ]
    for p in punct:
        text = text.replace(p, '')
    return text.strip()


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
    return [t.text for t in doc.noun_chunks]


def get_all_entity_map(entity_lists):
    res = []
    for entity_list in entity_lists:
        res.extend(entity_list)
    # return res, dict(zip(res, range(len(res))))
    return res


def analysis_data_file(in_path, out_path, err_path):
    with open(in_path, "r", encoding='utf-8') as f, open(out_path, 'a', encoding='utf-8') as f2, open(err_path, 'a', encoding='utf-8') as f3:
        count = 0
        captions = []
        datas = []
        for idx, line in enumerate(f):
            data = json.loads(line)
            if data['status'] == 'success':
                caption = data["caption"]
                captions.append(remove_punctuation(caption))
            datas.append(data)
            count += 1
            if count == 20:
                entity_lists = [get_entities(caption) for caption in captions]
                entity_offsets = [get_entity_offset(cap, entities) for cap, entities in zip(captions, entity_lists)]
                entity_offset_cont = []
                cur_offset = 0
                for offset_list in entity_offsets:
                    offsets = []
                    for offset in offset_list:
                        cur_offset += offset
                        offsets.append(cur_offset)
                    entity_offset_cont.append(offsets)
                all_entities = get_all_entity_map(entity_lists)
                assert len(entity_lists) == len(entity_offset_cont)
                all_idx = 0
                for i in range(len(entity_lists)):
                    groundings = {}
                    normal = True
                    original_groundings = {}
                    for j in range(len(entity_lists[i])):
                        old_entity = entity_lists[i][j]
                        offset = entity_offset_cont[i][j]
                        if old_entity in datas[i]['groundings']:
                            if all_idx+offset >= len(all_entities):
                                normal = False
                                all_idx += 1
                                continue
                            new_entity = all_entities[all_idx - offset]
                            groundings[new_entity] = datas[i]['groundings'][old_entity]
                            original_groundings[new_entity] = datas[i]['original_groundings'][old_entity]
                        all_idx += 1
                    # if idx == 19:
                    #     # print("entity_lists: ", entity_lists)
                    #     # print("entity_offsets: ", entity_offsets)
                    #     # print("all_entities: ", all_entities)
                    #     print("caption:", captions[i])
                    #     print("groundings:", groundings)
                    if normal:
                        datas[i]['groundings'] = groundings
                        datas[i]['original_groundings'] = original_groundings
                        f2.write(json.dumps(datas[i]) + '\n')
                    else:
                        f3.write(json.dumps(datas[i]) + '\n')
                # if idx == 19:
                #     print("entity_offset_cont:", entity_offset_cont)
                count = 0
                captions = []
                datas = []
        f2.close()
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--master_addr', type=str, default='')
    parser.add_argument('--master_port', type=int, default=7878)
    args = parser.parse_args()

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.cuda.set_device(args.local_rank)

    # spacy.prefer_gpu(args.local_rank)
    nlp = spacy.load("en_core_web_trf")
    input_path = "/nxchinamobile2/shared/jjh/laion115m_grounding"
    output_path = "/nxchinamobile2/shared/jjh/laion115m-debug/"
    process_list = []
    for dir_i, dir in enumerate(os.listdir(output_path)):
        print("processing dir {} {}/{}...".format(os.listdir(output_path), dir_i, len(os.listdir(output_path))))
        output_dir_path = os.path.join(output_path, dir)
        meta_dir_path = os.path.join(input_path, dir)
        files = os.listdir(meta_dir_path)
        for filename in tqdm(files):
            in_file_path = os.path.join(meta_dir_path, filename)
            out_file_path = os.path.join(output_dir_path, filename)
            err_file_path = os.path.join(output_dir_path, "error_"+filename)
            p = Process(target=analysis_data_file, args=(in_file_path, out_file_path, err_file_path))
            p.start()
            process_list.append(p)
            if len(process_list) >= 256:
                for p in process_list:
                    p.join()
                process_list = []

