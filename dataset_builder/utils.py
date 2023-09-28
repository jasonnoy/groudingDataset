import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import webdataset as wds


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
    per = []
    origin = []
    for idx, box in enumerate(pred.bbox):
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        if new_labels[idx] in new_to_old_entity:
            entity = new_to_old_entity[new_labels[idx]]
            pos = new_entity_to_id[new_labels[idx]][0]
            end = new_entity_to_id[new_labels[idx]][1]
        else:
            entity = new_labels[idx]
            pos = 0
            end = 0
            print("invalid entity:", entity)

        origin_loc = top_left+bottom_right
        loc = origin_loc

        if percent:
            width, height = pred.size
            loc = [l/width if i % 2 == 0 else l/height for i, l in enumerate(loc)]

        per.append([entity, pos, end] + loc)
        origin.append([entity, pos, end] + origin_loc)
    return per, origin


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


def imsave(img, caption, save_dir, index=None):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)
    if index is not None:
        caption=index
    plt.savefig(os.path.join(save_dir, "{}.png".format(caption)))
    plt.close()


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
