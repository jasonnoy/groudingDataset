from GLIP import *
import numpy as np
from PIL import Image


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


def output_decorator(bboxes, labels):
    res = []
    for idx, box in enumerate(bboxes):
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        res.append(top_left+bottom_right+[labels[idx]])
    return res


def load(path):
    pil_image = Image.open(path).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


if __name__ == "__main__":
    image = load('test.jpg')
    caption = 'bobble heads on top of the shelf'
    result, pred = glip_demo.run_on_web_image(image, caption, 0.5)
    bboxes = pred.bbox
    labels = get_label_names(pred, glip_demo)
    print(output_decorator(bboxes, labels))