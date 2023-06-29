import os

from PIL import Image
import numpy as np
from GLIP.maskrcnn_benchmark.config import cfg
from GLIP.maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
import sys

sys.path.append(os.path.join(os.getcwd(), "GLIP"))
sys.path.append("/home/jijunhui/projects/groudingDataset/GLIP/maskrcnn_benchmark")



config_file = "GLIP/configs/pretrain/glip_Swin_L.yaml"
weight_file = "/zphz/jjh/models/glip/MODEL/glip_large_model.pth"
cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])


import requests
from io import BytesIO
def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


glip_demo = GLIPDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
    show_mask_heatmaps=False
)

image = load('http://farm4.staticflickr.com/3693/9472793441_b7822c00de_z.jpg')
caption = 'bobble heads on top of the shelf'
result, _ = glip_demo.run_on_web_image(image, caption, 0.5)
print("result len:", len(result))
