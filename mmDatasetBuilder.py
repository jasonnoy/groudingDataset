from GLIP import *
import numpy as np
from PIL import Image
import spacy
import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pylab as pylab


def get_label_names(predictions, model):
    labels = predictions.get_field("labels").tolist()
    new_labels = []
    if model.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
        plus = 1
    else:
        plus = 0
    model.plus = plus
    if model.entities and model.plus:
        for i in labels:
            if i <= len(model.entities):
                new_labels.append(model.entities[i - model.plus])
            else:
                new_labels.append('object')
        # labels = [self.entities[i - self.plus] for i in labels ]
    else:
        new_labels = ['object' for i in labels]
    return new_labels


def get_grounding_and_label(pred, labels):
    res = defaultdict(list)
    for idx, box in enumerate(pred.bbox):
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        res[labels[idx]].append(top_left+bottom_right)
    return res


def output_decorator(id, caption, groundings, nouns, positions, subtexts, image_size):
    res = {"id": id, "caption": caption, "groundings": groundings, "nouns": nouns, "positions": positions,
           "subtexts": subtexts, "image_size": image_size}
    return res


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
        result, pred = glip_demo.run_on_image([image], [chunk_text], 0.55)
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
    res = output_decorator(idx, caption, total_groundings, nouns, ids, texts, image_size)
    return res


def parse_and_grounding_multi_class(img, caption, idx, nlp, output_path):
    image, image_size = load(img)
    doc = nlp(caption)
    nouns = []
    ids = []
    texts = []
    total_groundings = {}
    for noun_chunk in doc.noun_chunks:
        chunk_text = noun_chunk.text
        result, pred = glip_demo.run_on_image(image, chunk_text, 0.55)
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
    res = output_decorator(idx, caption, total_groundings, nouns, ids, texts, image_size)
    return res


if __name__ == "__main__":
    pylab.rcParams['figure.figsize'] = 20, 12
    nlp = spacy.load("en_core_web_trf")
    input_path = "/home/jijunhui/download/test_images"
    res = []
    with open(os.path.join(input_path, "meta.json"), 'r', encoding='utf-8') as f1:
        meta = json.loads(f1.read())
    f1.close()
    # doc = nlp(caption)
    # nouns = []
    # ids = []
    # texts = []
    # for noun_chunk in doc.noun_chunks:
    #     chunk_text = noun_chunk.text
    #     # chunk_text = 'bobble heads on top of the shelf'
    #     nouns.append(chunk_text)
    #     ids.append([t.idx for t in noun_chunk])
    #     texts.append(" ".join(t.text for t in noun_chunk.subtree))
    #     result, pred = glip_demo.run_on_web_image(image, chunk_text, 0.5)
    #     labels = get_label_names(pred, glip_demo)
    #     groundings = get_grounding_and_label(pred, labels)
    #     print("groundings:", groundings)
    # res = output_decorator(0, caption, groundings, nouns, ids, texts, image_size)
    for filename in tqdm(os.listdir(input_path)):
        if not filename.endswith(".png"):
            continue
        idx = filename.split(".")[0]
        output_path = os.path.join("output_1", str(idx))
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        caption = meta[str(idx)]['caption']
        ret = parse_and_grounding_single_class(os.path.join(input_path, filename), caption, str(idx), nlp, output_path)
        res.append(ret)
    with open("output_1/test.json", "w", encoding='utf-8') as f2:
        f2.write(json.dumps(res))
    f2.close()
