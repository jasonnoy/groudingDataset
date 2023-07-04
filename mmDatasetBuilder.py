from GLIP import *
import numpy as np
from PIL import Image
import spacy
import json
import os
import matplotlib.pyplot as plt


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
    res = []
    for idx, box in enumerate(pred.bbox):
        record = []
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        record.append(top_left+bottom_right)
        record.append(labels[idx])
        res.append(record)
    return res


def output_decorator(id, caption, groundings, nouns, positions, subtexts, image_size):
    res = {"id": id, "caption": caption, "groundings": groundings, "nouns": nouns, "positions": positions,
           "subtexts": subtexts, "image_size": image_size}
    return res


def load(path):
    pil_image = Image.open(path).convert("RGB")
    img_size = pil_image.size
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image, img_size


def imsave(img, caption):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)
    plt.savefig("{}.png".format(caption))


def parse_and_grounding_single_class(img, caption, id, nlp):
    image, image_size = load(img)
    doc = nlp(caption)
    nouns = []
    ids = []
    texts = []
    total_groundings = []
    for noun_chunk in doc.noun_chunks:
        chunk_text = noun_chunk.text
        nouns.append(chunk_text)
        ids.append([t.idx for t in noun_chunk])
        text = " ".join(t.text for t in noun_chunk.subtree)
        texts.append(text)
        result, pred = glip_demo.run_on_web_image(image, chunk_text, 0.55)
        print("chunk_text:", chunk_text)
        print("pred:", pred)
        labels = get_label_names(pred, glip_demo)
        groundings = get_grounding_and_label(pred, labels)
        total_groundings.extend(groundings)
        imsave(result, text)
    res = output_decorator(id, caption, total_groundings, nouns, ids, texts, image_size)
    return res


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_trf")
    img_path = 'test.jpg'
    caption = 'bobble heads on top of the shelf'
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
    res = parse_and_grounding_single_class(img_path, caption, 0, nlp)
    with open("test.json", "w", encoding='utf-8') as f:
        f.write(json.dumps(res))
    f.close()
