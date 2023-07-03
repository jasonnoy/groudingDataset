from GLIP import *
import numpy as np
from PIL import Image


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
    labels = pred.get_field("labels")
    print(output_decorator(bboxes, labels))