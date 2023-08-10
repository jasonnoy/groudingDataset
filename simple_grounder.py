import os
import torch
import spacy
import json
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "GLIP"))
from GLIP import *
from dataset_builder import *

config_file = "GLIP/configs/pretrain/glip_Swin_L.yaml"
weight_file = "/nxchinamobile2/shared/official_pretrains/hf_home/GLIP-L/glip_large_model.pth"

cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda:0"])
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.set_device(0)
cfg.MODEL.LANGUAGE_BACKBONE.LOCAL_PATH = "/nxchinamobile2/shared/official_pretrains/hf_home/bert-base-uncased"

nlp = spacy.load("en_core_web_trf")

glip_demo = GLIPDemo(
    cfg,
    confidence_threshold=0.7,
    show_mask_heatmaps=False,
    min_image_size=255,
    nlp=nlp
)


def build_temp_dataset(data_path, img_dir_path):
    img_filenames = os.listdir(img_dir_path)
    img_ids = [f.split(sep='.')[0] for f in img_filenames]
    dataset = []
    with open(data_path, "r", encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            key = data['key']
            if key not in img_ids:
                continue
            filename = img_filenames[img_ids.index(key)]
            with open(os.path.join(img_dir_path, filename), 'rb') as f2:
                b_img = f2.read()
            b_key = key.encode()
            b_caption = data['caption'].encode()
            dataset.append({'id': b_key, 'jpg': b_img, 'txt': b_caption})
    return dataset


if __name__ == "__main__":
    batch_size = 1
    output_dir_path = "/nxchinamobile2/shared/jjh/grit-dataset-20M/demo_100/vis_grit_exp"
    meta_path = "/nxchinamobile2/shared/jjh/grit-dataset-20M/demo_100/grit_coyo_100.jsonl"
    img_dir_path = "/nxchinamobile2/shared/jjh/grit-dataset-20M/demo_100/imgs"
    grit_dataset = build_temp_dataset(meta_path, img_dir_path)
    groundings = batch_parse_and_grounding_multi_class(glip_demo, grit_dataset, batch_size=batch_size,
                                                       output_path=output_dir_path, save_img=True)
    meta_filename = "grit_grounding_100.jsonl"
    output_meta_path = os.path.join(output_dir_path, meta_filename)
    if os.path.exists(output_meta_path):
        os.remove(output_meta_path)
    with open(output_meta_path, 'a', encoding='utf-8') as f2:
        for grounding in groundings:
            f2.write(json.dumps(grounding, ensure_ascii=False) + '\n')
    f2.close()
