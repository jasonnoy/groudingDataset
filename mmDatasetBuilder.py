import torch.utils.data
from GLIP.maskrcnn_benchmark.data.datasets.laion import Laion
from GLIP import *
import numpy as np
from PIL import Image
import spacy
import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import io
import webdataset as wds
import time
from transformers import AutoTokenizer


def get_label_names(predictions, model):
    labels = predictions.get_field("labels").tolist()
    new_labels = []
    if model.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
        plus = 1
    else:
        plus = 0
    model.plus = plus
    if model.new_entities and model.plus:
        for i in labels:
            if i <= len(model.new_entities):
                new_labels.append(model.new_entities[i - model.plus])
            else:
                new_labels.append('object')
        # labels = [self.entities[i - self.plus] for i in labels ]
    else:
        new_labels = ['object' for i in labels]
    return new_labels


def get_grounding_and_label(pred, new_labels, new_entity_to_id, new_to_old_entity):
    res = defaultdict(list)
    for idx, box in enumerate(pred.bbox):
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        if new_labels[idx] in new_to_old_entity:
            entity = new_to_old_entity[new_labels[idx]]
            pos = new_entity_to_id[new_labels[idx]]
        else:
            entity = new_labels[idx]
            pos = -1
        res[entity].append([top_left+bottom_right, pos])
    return res


# def output_decorator(id, caption, groundings, image_size):
#     res = {"id": id, "caption": caption, "groundings": groundings, "image_size": image_size}
#     return res


def output_decorator(groundings):
    res = {"groundings": groundings}
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
    dataloader = torch.utils.data.DataLoader(laion_dataset, shuffle=False, num_workers=4, batch_size=batch_size)
    total_groundings = []
    for batch in tqdm(dataloader):
        # print("batch:", batch)
        # print(*batch[:, :3])

        results, preds = glip_demo.run_on_batched_images(*batch[:3], thresh=0.55, save_img=save_img)
        new_to_old_entities = batch[3]
        new_entity_to_ids = batch[4]
        for result, pred, new_entity_to_id, new_to_old_entity in zip(results, preds, new_entity_to_ids, new_to_old_entities):
            new_labels = get_label_names(pred, glip_demo)
            if save_img:
                result = glip_demo.overlay_entity_names(result, pred, custom_labels=new_labels, text_size=0.8,
                                                        text_offset=-25,
                                                        text_offset_original=-40, text_pixel=2)
                imsave(result, caption, output_path)
            groundings = get_grounding_and_label(pred, new_labels, new_entity_to_id, new_to_old_entity)
            total_groundings.append(output_decorator(groundings))
    return total_groundings


def read_tar(tar_path):
    return wds.WebDataset(tar_path)


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_trf")
    input_path = "/gpfs/gpfs1/zphz/img_datasets/laion115m/part-00033"
    output_path = "/gpfs/gpfs1/zphz/jjh/test_dataset/part-00033"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    ids = [0, 1, 2, 3, 4]
    part_index = 3300000
    for idx in ids:
        res = {}
        # tar_filename = "{}.tar".format(part_index+idx)
        # tar_dataset = read_tar(os.path.join(input_path, tar_filename))
        tokenizer = AutoTokenizer.from_pretrained("/gpfs/gpfs1/zphz/official_pretrains/hugging_face/bert-base-uncased")
        laion_dataset = Laion(str(part_index+idx), input_path, nlp, tokenizer)
        print(laion_dataset[0])
        meta_filename = "{}.meta.jsonl".format(part_index+idx)
        print("processing {}".format(part_index+idx))
        groundings = batch_parse_and_grounding_multi_class(laion_dataset, batch_size=10, save_img=False, output_path=output_path)
        print("grounding 0: ", groundings[0])
        print("groundings size:", len(groundings))
        break
        with open(os.path.join(input_path, meta_filename), 'r', encoding='utf-8') as f1, open(os.path.join(output_path, meta_filename), 'a', encoding='utf-8') as f2:
            # for data, line in tqdm(zip(tar_dataset, f1)):
            iter_tar = iter(tar_dataset)
            for i, line in tqdm(enumerate(f1)):
                meta_data = json.loads(line)
                if meta_data['status'] == "success":
                    data = next(iter_tar)
                    size = (int(meta_data['width']), int(meta_data['height']))
                    index = data['id'].decode()
                    sample_id = meta_data['SAMPLE_ID']
                    if str(index) != str(sample_id):
                        print("index:{}\n sample_id:{}".format(str(index), str(sample_id)))
                    image_b = data['jpg']
                    image = Image.open(io.BytesIO(image_b)).convert('RGB')
                    caption = data['txt'].decode()
                    ret = parse_and_grounding_multi_class(image, caption, str(idx), nlp, output_path, i < 5)  # save first 5 grounding images for each tar
                    meta_data.update(ret)
                else:
                    meta_data['grounding'] = None
                f2.write(json.dumps(meta_data, ensure_ascii=False) + '\n')
                break
        f1.close()
        f2.close()
        break
    print("done")
