import argparse
import torch
import json
import spacy
from transformers import AutoTokenizer
from tqdm import tqdm
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

    config_file = "../GLIP/configs/pretrain/glip_Swin_L.yaml"
    weight_file = "/gpfs/gpfs1/zphz/jjh/models/glip/MODEL/glip_large_model.pth"
    cfg.local_rank = args.local_rank
    cfg.num_gpus = 1
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda:{}".format(args.local_rank)])
    torch.cuda.set_device(args.local_rank)
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
    input_path = "/gpfs/gpfs1/zphz/img_datasets/laion115m/part-00032"
    output_path = "/gpfs/gpfs1/zphz/jjh/test_dataset/part-00032"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    total_ids = get_id_list(input_path)
    part_size = len(total_ids) // max((world_size-1), 1)
    start = rank*part_size
    end = min((rank+1)*part_size, len(total_ids))
    ids = total_ids[start:end]
    for idx in ids:
        idx = int(idx)
        res = {}
        tokenizer = AutoTokenizer.from_pretrained("/gpfs/gpfs1/zphz/official_pretrains/hugging_face/bert-base-uncased")
        batch_size = 5
        laion_dataset = Laion(str(idx), input_path, nlp, tokenizer, transforms=glip_demo.transforms)
        meta_filename = "{}.meta.jsonl".format(idx)
        print("processing {}".format(idx))
        groundings = batch_parse_and_grounding_multi_class(laion_dataset, batch_size=batch_size, save_img=False, output_path=output_path)
        output_meta_path = os.path.join(output_path, meta_filename)
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
