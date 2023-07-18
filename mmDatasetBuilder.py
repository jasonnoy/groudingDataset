import argparse
import torch.utils.data
from GLIP import *
from GLIP.maskrcnn_benchmark.data.datasets.laion import Laion
from GLIP.maskrcnn_benchmark.data.collate_batch import BatchGroundingCollator
from GLIP.maskrcnn_benchmark.config import cfg
from GLIP.maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
import numpy as np
from PIL import Image
import spacy
import json
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import io
import webdataset as wds
import time
from transformers import AutoTokenizer
from functools import reduce
from operator import add


def get_label_names(predictions, model, new_entities):
    labels = predictions.get_field("labels").tolist()
    new_labels = []
    if model.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
        plus = 1
    else:
        plus = 0
    model.plus = plus
    if plus:
        for i in labels:
            if i <= len(new_entities):
                new_labels.append(new_entities[i - model.plus])
            else:
                new_labels.append('object')
        # labels = [self.entities[i - self.plus] for i in labels ]
    else:
        new_labels = ['object' for i in labels]
    return new_labels


def get_grounding_and_label(pred, new_labels, new_entity_to_id, new_to_old_entity, percent=False):
    res = {}
    for idx, box in enumerate(pred.bbox):
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        if new_labels[idx] in new_to_old_entity:
            entity = new_to_old_entity[new_labels[idx]]
            pos = new_entity_to_id[new_labels[idx]]
        else:
            entity = new_labels[idx]
            pos = -1

        loc = top_left+bottom_right

        if percent:
            width, height = pred.size
            loc = [l/width if l % 2 == 0 else l/height for l in loc]

        if entity not in res:
            res[entity] = {pos: [loc]}
        elif pos in res[entity]:
            res[entity][pos].append(loc)
        else:
            res[entity][pos] = [loc]
    return res


# def output_decorator(id, caption, groundings, image_size):
#     res = {"id": id, "caption": caption, "groundings": groundings, "image_size": image_size}
#     return res


def output_decorator(groundings, idx):
    res = {"groundings": groundings, "SAMPLE_ID": idx}
    return res


def load_img(pil_image):
    # if img_size != (640, 480):
    #     pil_image = pil_image.resize((640, 480))
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


def load(path):
    pil_image = Image.open(path).convert("RGB")
    img_size = pil_image.size
    if img_size != (640, 480):
        pil_image = pil_image.resize((640, 480))
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image, img_size


def imsave(img, caption, save_dir):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)
    plt.savefig(os.path.join(save_dir, "{}.png".format(caption)))
    plt.close()


def parse_and_grounding_single_class(img, caption, idx, nlp, output_path):
    image, image_size = load(img)
    doc = nlp(caption)
    nouns = []
    ids = []
    texts = []
    total_groundings = {}
    for noun_chunk in doc.noun_chunks:
        chunk_text = noun_chunk.text
        result, pred = glip_demo.run_on_image(image, chunk_text, 0.55)
        # print("result:", result)
        # print("pred:", pred)
        # if no detection
        if len(pred.bbox) == 0:
            continue
        nouns.append(chunk_text)
        ids.append([t.idx for t in noun_chunk])
        text = " ".join(t.text for t in noun_chunk.subtree)
        texts.append(text)
        image_size = pred.size
        labels = get_label_names(pred, glip_demo)
        labels = [text] * len(labels)
        result = glip_demo.overlay_entity_names(result, pred, custom_labels=labels, text_size=0.8, text_offset=-25, text_offset_original=-40, text_pixel=2)
        groundings = get_grounding_and_label(pred, labels)
        total_groundings.update(groundings)
        imsave(result, text, output_path)
    res = output_decorator(total_groundings)
    return res


def parse_and_grounding_multi_class(img, caption, idx, nlp, output_path, save_img=False):
    image = load_img(img)
    # image = np.array(img)[:, :, [2, 1, 0]]
    doc = nlp(caption)
    nouns = [t.text.lower() for t in doc.noun_chunks]
    empty_nouns = False
    if len(nouns) == 0:
        print("No entities found, using caption as entity, caption: {}".format(caption))
        nouns = [caption.lower()]
        empty_nouns = True
    entity_dict = {}
    new_entities = []
    # to handle duplicates in entities
    for chunk in nouns:
        if chunk not in entity_dict:
            new_entities.append(chunk)
            entity_dict[chunk] = 0
        else:
            entity_dict[chunk] += 1
            new_entities.append("{}-{}".format(chunk, entity_dict[chunk]))
    glip_demo.new_entities = new_entities
    new_to_old_entity = dict(zip(new_entities, nouns))
    if not empty_nouns:
        new_entity_to_id = dict(zip(new_entities, [noun_chunk[0].idx for noun_chunk in doc.noun_chunks]))  # starting position of the first token
    else:
        # use caption as only entity
        new_entity_to_id = {new_entities[0]: 0}
    total_groundings = {}
    result, pred = glip_demo.run_on_image(image, caption, 0, custom_entity=nouns, save_img=save_img)

    new_labels = get_label_names(pred, glip_demo)
    # print("labels:", labels)
    if save_img:
        result = glip_demo.overlay_entity_names(result, pred, custom_labels=new_labels, text_size=0.8, text_offset=-25,
                                                text_offset_original=-40, text_pixel=2)
        imsave(result, caption, output_path)

    groundings = get_grounding_and_label(pred, new_labels, new_entity_to_id, new_to_old_entity)
    total_groundings.update(groundings)
    # for noun_chunk in doc.noun_chunks:
    #     chunk_text = noun_chunk.text
    #     # if no detection
    #     if len(pred.bbox) == 0:
    #         continue
    #     nouns.append(chunk_text)
    #     ids.append([t.idx for t in noun_chunk])
    #     text = " ".join(t.text for t in noun_chunk.subtree)
    #     texts.append(text)

    res = output_decorator(total_groundings)
    return res


def batch_parse_and_grounding_multi_class(laion_dataset, batch_size, output_path, save_img=False):
    dataloader = torch.utils.data.DataLoader(laion_dataset, shuffle=False, num_workers=2, batch_size=batch_size, collate_fn=BatchGroundingCollator())
    total_groundings = []
    for i, batch in tqdm(enumerate(dataloader)):
        origin_images = batch[7]
        results, preds = glip_demo.run_on_batched_images(*batch[:5], origin_images=origin_images, thresh=0.55, save_img=save_img)
        captions = batch[2]
        new_entities = batch[4]
        new_to_old_entities = batch[5]
        new_entity_to_ids = batch[6]
        image_ids = batch[8]
        entire_entities = reduce(add, new_entities)
        if results:
            for result, pred, caption, new_entity_to_id, new_to_old_entity, index in zip(results, preds, captions, new_entity_to_ids, new_to_old_entities, image_ids):
                new_labels = get_label_names(pred, glip_demo, entire_entities)
                old_labels = [new_to_old_entity[label] for label in new_labels]
                if save_img:
                    result = glip_demo.overlay_entity_names(result, pred, entire_entities, custom_labels=old_labels, text_size=0.8,
                                                            text_offset=-25,
                                                            text_offset_original=-40, text_pixel=2)
                    imsave(result, caption, output_path)
                groundings = get_grounding_and_label(pred, new_labels, new_entity_to_id, new_to_old_entity)
                total_groundings.append(output_decorator(groundings, index))
        else:
            for pred, new_entity_to_id, new_to_old_entity, index in zip(preds, new_entity_to_ids, new_to_old_entities, image_ids):
                new_labels = get_label_names(pred, glip_demo, entire_entities)
                groundings = get_grounding_and_label(pred, new_labels, new_entity_to_id, new_to_old_entity)
                total_groundings.append(output_decorator(groundings, index))
    return total_groundings


def read_tar(tar_path):
    return wds.WebDataset(tar_path)


def get_relative_coords_str(coords, width, height):
    x_1 = round(coords[0] / width, 3)
    y_1 = round(coords[1] / height, 3)
    x_2 = round(coords[2] / width, 3)
    y_2 = round(coords[3] / height, 3)
    return "{},{},{},{}".format(x_1, y_1, x_2, y_2)


# def sort_groundings(groundings: list):


def build_training_text(record):
    width = record['width']
    height = record['height']
    caption = list(record['caption'])
    groundings = record['groundings'].values()
    loc_pos_list = []
    for g in groundings:
        for pos, locs in g.items():
            loc_pos_list.append([locs, pos])
    # print("loc_pos_list:", loc_pos_list)
    sorted_groundings = sorted(loc_pos_list, key=lambda x: x[1], reverse=True)
    for grounding_pair in sorted_groundings:
        locs = grounding_pair[0]
        pos = grounding_pair[1]
        loc_strs = [get_relative_coords_str(coords, width, height) for coords in locs]
        grouning_str = ";".join(loc_strs)
        grouning_str = "[{}]".format(grouning_str)
        caption.insert(pos, grouning_str)
    caption = "".join(caption)
    # print("caption:", caption)
    return caption


def get_id_list(path):
    filenames = os.listdir(path)
    id_list = [name[:-4] for name in filenames if name.endswith('.tar')]
    return id_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--master_addr', type=str, default='')
    parser.add_argument('--master_port', type=int, default=7878)
    args = parser.parse_args()

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    config_file = "./GLIP/configs/pretrain/glip_Swin_L.yaml"
    weight_file = "/gpfs/gpfs1/zphz/jjh/models/glip/MODEL/glip_large_model.pth"
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

    glip_demo = GLIPDemo(
        cfg,
        confidence_threshold=0.7,
        show_mask_heatmaps=False,
        min_image_size=800
    )

    nlp = spacy.load("en_core_web_trf")
    input_path = "/gpfs/gpfs1/zphz/img_datasets/laion115m/part-00032"
    output_path = "/gpfs/gpfs1/zphz/jjh/test_dataset/part-00032"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    total_ids = get_id_list(input_path)
    part_size = len(total_ids) // max((world_size-1), 1)
    start = rank*part_size
    end = min((rank+1)*part_size, len(total_ids))
    ids = total_ids[start:end]
    for idx in ids:
        idx = int(idx)
        res = {}
        # tar_filename = "{}.tar".format(part_index+idx)
        # tar_dataset = read_tar(os.path.join(input_path, tar_filename))
        tokenizer = AutoTokenizer.from_pretrained("/gpfs/gpfs1/zphz/official_pretrains/hugging_face/bert-base-uncased")
        batch_size = 5
        laion_dataset = Laion(str(idx), input_path, nlp, tokenizer, transforms=glip_demo.transforms)
        meta_filename = "{}.meta.jsonl".format(idx)
        print("processing {}".format(idx))
        groundings = batch_parse_and_grounding_multi_class(laion_dataset, batch_size=batch_size, save_img=False, output_path=output_path)
        output_meta_path = os.path.join(output_path, meta_filename)
        if os.path.exists(output_meta_path):
            os.remove(output_meta_path)
        with open(os.path.join(input_path, meta_filename), 'r', encoding='utf-8') as f1, open(output_meta_path, 'a', encoding='utf-8') as f2:
            grounding_iter = iter(groundings)
            for i, line in tqdm(enumerate(f1)):
                meta_data = json.loads(line)
                if meta_data['status'] == "success":
                    grounding = next(grounding_iter)
                    # size = (int(meta_data['width']), int(meta_data['height']))
                    # index = data['id'].decode()
                    image_id = grounding['SAMPLE_ID']
                    sample_id = meta_data['SAMPLE_ID']
                    if str(image_id) != str(sample_id):
                        print("index:{}\n sample_id:{}".format(str(image_id), str(sample_id)))
                    meta_data.update(grounding)
                    meta_data['annot_caption'] = build_training_text(record=meta_data)
                else:
                    meta_data['grounding'] = None
                    meta_data['annot_caption'] = None
                    loc_pos_list = None
                f2.write(json.dumps(meta_data, ensure_ascii=False) + '\n')
        f1.close()
        f2.close()
    print("done")
