import argparse
import re
import torch
import json
import os
import sys
import spacy
from tqdm import tqdm
import warnings
import math
warnings.filterwarnings("ignore")

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "GLIP"))

from GLIP import *
from dataset_builder import *

def is_small_object(json_object, thresh=0.1):
    for item, groundings in dict(json_object['task_data']['groundings']).items():
        for pos, boxes in groundings.items():
            for box in boxes:
                x_1, y_1, x_2, y_2 = box
                if (x_2 - x_1) * (y_2 - y_1) < thresh:
                    return True
    return False


def split_list_by_n(origin_list, n):
    step = math.ceil(len(origin_list) / n)
    res = []
    for i in range(0, len(origin_list), step):
        res.append(origin_list[i:i + step])
    return res


def get_durations(j_caption, start=0):
    durations = []
    end = len(j_caption)
    for it in re.finditer(r'\|\|', j_caption):
        breaker = it.start()
        durations.append((start, breaker))
        start = breaker
    durations.append((start, end))
    return durations


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--master_addr', type=str, default='')
    parser.add_argument('--master_port', type=int, default=7878)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--thresh', type=float, default=0.55)
    parser.add_argument('--save_img', action='store_true')
    args = parser.parse_args()

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    config_file = "GLIP/configs/pretrain/glip_Swin_L.yaml"
    weight_file = "/share/official_pretrains/hf_home/GLIP-L/glip_large_model.pth"
    cfg.local_rank = args.local_rank
    cfg.num_gpus = 1
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda:{}".format(args.local_rank)])
    torch.cuda.set_device(args.local_rank)
    cfg.MODEL.LANGUAGE_BACKBONE.LOCAL_PATH = "/share/official_pretrains/hf_home/bert-base-uncased"
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
        min_image_size=255,
        nlp=nlp
    )

    input_path = "/share/img_datasets/instruction_data/CoM"
    output_path = "/share/img_datasets/instruction_data/CoM_Grounding"

    all_paths = []
    for dir in os.listdir(input_path):
        if dir == "GQA":
            continue
        out_dir_path = os.path.join(output_path, dir)
        os.makedirs(out_dir_path, exist_ok=True)
        for file in os.listdir(os.path.join(input_path, dir)):
            if file.endswith(".jsonl"):
                file_path = os.path.join(input_path, dir, file)
                all_paths.append(file_path)

    divided_paths = split_list_by_n(all_paths, world_size)
    select_paths = divided_paths[rank]

    for meta_path in select_paths:
        com_dataset = []
        joint_captions = []
        with open(meta_path, "r") as f:
            for i, l in enumerate(f):
                data = json.loads(l)
                img_path = data['image_path']
                questions = []
                for qa in data['metadata']:
                    if "question" in qa:
                        questions.append(qa['question'])
                    elif "prompt" in qa:
                        questions.append(qa['prompt'])
                    else:
                        raise Exception(f"No question in metadata: {data}")
                caption = "||".join(questions)
                joint_captions.append(caption)
                com_dataset.append({"image_path": img_path, "caption": caption, "id": "%05d" % i})

        output_meta_path = meta_path.replace(input_path, output_path)
        output_img_path = os.path.dirname(output_meta_path)
        batch_size = args.batch_size

        res = {}

        # try:
        groundings = batch_parse_and_grounding_multi_class(glip_demo, com_dataset, batch_size=batch_size, output_path=output_img_path, save_img=args.save_img, use_decor=False, thresh=args.thresh)
        # except Exception as e:
        #     print("failed batch_parse_and_grounding_multi_class for {}, skipping...".format(os.path.join(input_dir_path, tar_file)))
        #     continue

        with open(meta_path, 'r', encoding='utf-8') as f1, open(output_meta_path, 'w', encoding='utf-8') as f2:
            for d_i, (line, grounding) in enumerate(zip(f1, groundings)):
                percent_grounding, index, origin_grounding = grounding
                data = json.loads(line)
                for d in data['metadata']:
                    d['question_grounding'] = []
                    d['question_grounding_percent'] = []
                joint_caption = joint_captions[d_i]
                durations = get_durations(joint_caption)
                for gr_i, gr in enumerate(percent_grounding):
                    start = int(gr[1])
                    end = int(gr[2])
                    for i, dura in enumerate(durations):
                        if dura[0] <= start <= dura[1]:
                            try:
                                assert dura[0] <= end <= dura[1]
                            except Exception:
                                print(dura, gr)
                                print(joint_caption)
                                continue
                            gr[1] -= dura[0]
                            gr[2] -= dura[0]
                            origin_grounding[gr_i][1] -= dura[0]
                            origin_grounding[gr_i][2] -= dura[0]
                            data['metadata'][i].update({"question_grounding": origin_grounding[gr_i], "question_grounding_percent": gr})
                f2.write(json.dumps(data, ensure_ascii=False) + '\n')
        f1.close()
        f2.close()
    print("done for node", rank)
