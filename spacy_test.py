import argparse
import torch
import json
import os
import re
import spacy
from tqdm import tqdm
from multiprocessing import Process
import math


def split_list_by_n(origin_list, n):
    step = math.ceil(len(origin_list) / n)
    res = []
    for i in range(0, len(origin_list), step):
        res.append(origin_list[i:i + step])
    return res


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


def get_entities(text, nlp):
    doc = nlp(text)
    return [t.text for t in doc.noun_chunks]


def get_all_entity_map(entity_lists):
    res = []
    for entity_list in entity_lists:
        res.extend(entity_list)
    # return res, dict(zip(res, range(len(res))))
    return res


def analysis_data_file(in_path, out_path, err_path, nlp):
    with open(in_path, "r", encoding='utf-8') as f, open(out_path, 'a', encoding='utf-8') as f2, open(err_path, 'a', encoding='utf-8') as f3:
        count = 0
        captions = []
        datas = []
        for idx, line in enumerate(f):
            print(idx)
            data = json.loads(line)
            if data['status'] == 'success':
                caption = data["caption"]
                captions.append(remove_punctuation(caption))
            datas.append(data)
            count += 1
            if count == 20:
                entity_lists = [get_entities(caption, nlp) for caption in captions]
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
                    if normal:
                        datas[i]['groundings'] = groundings
                        datas[i]['original_groundings'] = original_groundings
                        f2.write(json.dumps(datas[i]) + '\n')
                    else:
                        f3.write(json.dumps(datas[i]) + '\n')
                count = 0
                captions = []
                datas = []
    f2.close()
    f3.close()
    f.close()


if __name__ == "__main__":
    # from spacy_cleaner.processing.replacers import replace_punctuation_token

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("merge_noun_chunks")
    nlp.add_pipe("merge_entities")
    for t in nlp("A group of people. Tim Cook would like a cup of coffee. But there's no one in the room."):
        print("text:", t.text, "pos:", t.pos_)
        print("head:", t.head)
        for an in t.ancestors:
            print("ancestor:", an.text)
