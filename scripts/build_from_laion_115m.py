import argparse
import torch
import json
import os
import sys
import spacy
from tqdm import tqdm
import webdataset
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "GLIP"))

from GLIP import *
from dataset_builder import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--master_addr', type=str, default='')
    parser.add_argument('--master_port', type=int, default=7878)
    args = parser.parse_args()

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    config_file = "GLIP/configs/pretrain/glip_Swin_L.yaml"
    weight_file = "/nxchinamobile2/shared/official_pretrains/hf_home/GLIP-L/glip_large_model.pth"
    cfg.local_rank = args.local_rank
    cfg.num_gpus = 1
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda:{}".format(args.local_rank)])
    torch.cuda.set_device(args.local_rank)
    cfg.MODEL.LANGUAGE_BACKBONE.LOCAL_PATH = "/nxchinamobile2/shared/official_pretrains/hf_home/bert-base-uncased"
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
    input_path = "/nxchinamobile2/shared/img_datasets/laion115m"
    output_path = "/nxchinamobile2/shared/jjh/laion115m"
    map_name = "file_map_laion_synthetic_filtered_large.json"
    map_key = "laion_synthetic_filtered_large.json"
    with open(os.path.join(input_path, map_name), 'r', encoding='utf-8') as f:
        data = f.read()
        data = json.loads(data)
    f.close()
    dirs = data[map_key]
    dir_size = len(dirs) // max((world_size//8-1), 1)
    dir_start = rank*dir_size
    dir_end = (rank+1)*dir_start
    select_dirs = dirs[dir_start:dir_end]
    for cur_dir in select_dirs:
        output_dir_path = os.path.join(output_path, str(cur_dir))
        input_dir_path = os.path.join(input_path, str(cur_dir))

        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)
        tar_files = get_id_list(input_dir_path)
        for tar_file in tar_files:
            idx = int(tar_file[:-4])
            res = {}
            batch_size = 10
            laion_dataset = webdataset.WebDataset(os.path.join(input_dir_path, tar_file))
            meta_filename = "{}.meta.jsonl".format(idx)
            print("processing {}".format(idx))
            groundings = batch_parse_and_grounding_multi_class(glip_demo, laion_dataset, batch_size=batch_size, save_img=False, output_path=output_dir_path)
            output_meta_path = os.path.join(output_dir_path, meta_filename)
            if os.path.exists(output_meta_path):
                os.remove(output_meta_path)
            with open(os.path.join(input_path, meta_filename), 'r', encoding='utf-8') as f1, open(output_meta_path, 'a', encoding='utf-8') as f2:
                grounding_iter = iter(groundings)
                for i, line in tqdm(enumerate(f1)):
                    meta_data = json.loads(line)
                    if meta_data['status'] == "success":
                        grounding = next(grounding_iter)
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
