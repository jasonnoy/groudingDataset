import torch.utils.data
from GLIP.maskrcnn_benchmark.data.collate_batch import BatchGroundingCollator
import numpy as np
from PIL import Image
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import webdataset as wds
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
    origin = {}
    for idx, box in enumerate(pred.bbox):
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        if new_labels[idx] in new_to_old_entity:
            entity = new_to_old_entity[new_labels[idx]]
            pos = new_entity_to_id[new_labels[idx]]
        else:
            entity = new_labels[idx]
            pos = -1

        origin_loc = top_left+bottom_right
        loc = origin_loc

        if percent:
            width, height = pred.size
            loc = [l/width if i % 2 == 0 else l/height for i, l in enumerate(loc)]

        if entity not in res:
            res[entity] = {pos: [loc]}
            origin[entity] = {pos: [origin_loc]}
        elif pos in res[entity]:
            res[entity][pos].append(loc)
            origin[entity][pos].append(origin_loc)
        else:
            res[entity][pos] = [loc]
            origin[entity][pos] = [origin_loc]
    return res, origin


# def output_decorator(id, caption, groundings, image_size):
#     res = {"id": id, "caption": caption, "groundings": groundings, "image_size": image_size}
#     return res


def output_decorator(groundings, idx, original_groundings=None):
    res = {"groundings": groundings, "SAMPLE_ID": idx, "original_groundings": original_groundings}
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


def batch_parse_and_grounding_multi_class(glip_demo, laion_dataset, batch_size, output_path, save_img=False):
    dataloader = torch.utils.data.DataLoader(laion_dataset, shuffle=False, num_workers=4, batch_size=batch_size,
                                             collate_fn=BatchGroundingCollator(glip_demo.nlp, glip_demo.tokenizer, glip_demo.transforms))
    total_groundings = []
    for i, batch in tqdm(enumerate(dataloader)):
        if i == 2:
            break
        origin_images = batch[7]
        # try:
        results, preds = glip_demo.run_on_batched_images(*batch[:5], origin_images=origin_images, thresh=0.55, save_img=save_img)
        # except Exception as e:
        #     print(e)
        #     for i in range(len(batch[8])):
        #         total_groundings.append(output_decorator(None, batch[8][i]))
        #     continue
        new_entities = batch[4]
        entire_entities = reduce(add, new_entities)
        new_to_old_entities = batch[5]
        new_entity_to_ids = batch[6]
        image_ids = batch[8]
        captions = batch[9]
        if results:
            for result, pred, caption, new_entity_to_id, new_to_old_entity, index in zip(results, preds, captions, new_entity_to_ids, new_to_old_entities, image_ids):
                new_labels = get_label_names(pred, glip_demo, entire_entities)
                old_labels = [new_to_old_entity[label] for label in new_labels]
                if save_img:
                    result = glip_demo.overlay_entity_names(result, pred, entire_entities, custom_labels=old_labels, text_size=0.8,
                                                            text_offset=-25,
                                                            text_offset_original=-40, text_pixel=2)
                    imsave(result, caption, output_path)
                groundings, origin_groundings = get_grounding_and_label(pred, new_labels, new_entity_to_id, new_to_old_entity, percent=True)
                total_groundings.append(output_decorator(groundings, index, origin_groundings))
        else:
            for pred, new_entity_to_id, new_to_old_entity, index in zip(preds, new_entity_to_ids, new_to_old_entities, image_ids):
                new_labels = get_label_names(pred, glip_demo, entire_entities)
                groundings, origin_groundings = get_grounding_and_label(pred, new_labels, new_entity_to_id, new_to_old_entity, percent=True)
                total_groundings.append(output_decorator(groundings, index, origin_groundings))
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
    id_list = [name for name in filenames if name.endswith('.tar')]
    id_list.sort()
    return id_list
