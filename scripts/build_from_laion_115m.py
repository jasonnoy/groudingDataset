import argparse
import torch
import json
import os
import sys
import spacy
from tqdm import tqdm
import webdataset
import warnings
import math
warnings.filterwarnings("ignore")

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "GLIP"))

from GLIP import *
from dataset_builder import *


def split_list_by_n(origin_list, n):
    step = math.ceil(len(origin_list) / n)
    res = []
    for i in range(0, len(origin_list), step):
        res.append(origin_list[i:i + step])
    return res


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--master_addr', type=str, default='')
    parser.add_argument('--master_port', type=int, default=7878)
    args = parser.parse_args()

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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
    local_rank = args.local_rank
    world_size = args.world_size

    # spacy.prefer_gpu(args.local_rank)
    nlp = spacy.load("en_core_web_trf")

    glip_demo = GLIPDemo(
        cfg,
        confidence_threshold=0.7,
        show_mask_heatmaps=False,
        min_image_size=800,
        nlp=nlp
    )

    input_path = "/nxchinamobile2/shared/img_datasets/laion115m"
    output_path = "/nxchinamobile2/shared/jjh/laion115m"
    map_name = "file_map_laion_synthetic_filtered_large.json"
    map_key = "laion_synthetic_filtered_large.json"
    with open(os.path.join(input_path, map_name), 'r', encoding='utf-8') as f:
        data = f.read()
        data = json.loads(data)
    f.close()
    dirs = data[map_key]
    id_list = []
    for dir in dirs:
        id_list.extend([file[:-4] for file in os.listdir(os.path.join(input_path, dir)) if file.endswith(".tar")])
    id_list.sort()
    divided_ids = split_list_by_n(id_list, world_size)
    select_ids = divided_ids[rank]
    print("node", rank, "select_ids:", select_ids)
    for cur_id in select_ids:
        cur_dir = "part-000{}".format(cur_id[:2])
        output_dir_path = os.path.join(output_path, str(cur_dir))
        input_dir_path = os.path.join(input_path, str(cur_dir))

        skip_ids = os.listdir(output_dir_path)
        skip_ids = [skip_id.split(sep='.')[0] for skip_id in skip_ids]
        if cur_id in skip_ids:
            print("rank {}, skip_id:".format(rank), cur_id)
            continue

        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)

        # print("rank {}, selected_tar_files:".format(rank), select_tar_files)
        tar_file = "{}.tar".format(cur_id)
        # if idx in skip_ids:
        #     print("skipping finished id", idx)
        #     continue
        res = {}
        batch_size = 20
        try:
            laion_dataset = webdataset.WebDataset(os.path.join(input_dir_path, tar_file))
        except Exception as e:
            print("failed to load dataset for {}, skipping...".format(os.path.join(input_dir_path, tar_file)))
            continue
        meta_filename = "{}.meta.jsonl".format(cur_id)
        print("rank {}, processing {}".format(rank, cur_id))
        groundings = batch_parse_and_grounding_multi_class(glip_demo, laion_dataset, batch_size=batch_size, save_img=False, output_path=output_dir_path)
        output_meta_path = os.path.join(output_dir_path, meta_filename)
        if os.path.exists(output_meta_path):
            os.remove(output_meta_path)
        with open(os.path.join(input_dir_path, meta_filename), 'r', encoding='utf-8') as f1, open(output_meta_path, 'a', encoding='utf-8') as f2:
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
                    # meta_data['annot_caption'] = build_training_text(record=meta_data)
                else:
                    meta_data['groundings'] = None
                    meta_data['original_groundings'] = None
                    # meta_data['annot_caption'] = None
                    loc_pos_list = None
                f2.write(json.dumps(meta_data, ensure_ascii=False) + '\n')
        f1.close()
        f2.close()
    print("done for node", rank)
