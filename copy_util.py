import argparse
import os
import json
from tqdm import tqdm
from multiprocessing import Process
import webdataset
import math


def split_list_by_n(origin_list, n):
    step = math.ceil(len(origin_list) / n)
    res = []
    for i in range(0, len(origin_list), step):
        res.append(origin_list[i:i + step])
    return res


def combine_files(in_path, meta_path, correct_path, out_path, backup_path, record_path):
    data = webdataset.WebDataset(in_path)
    backup_dict = {}
    with open(backup_path, 'r', encoding='utf-8') as f:
        for line in f:
            back_data = json.loads(line)
            backup_dict[back_data['SAMPLE_ID']] = back_data
    with open(correct_path, 'r', encoding='utf-8') as f1, open(meta_path, 'r', encoding='utf-8') as f2, open(out_path, 'w', encoding='utf-8') as f3, open(record_path, 'a', encoding='utf-8') as f4:
        res_dict = {}
        for f1_line in f1:
            correct_data = json.loads(f1_line)
            res_dict[correct_data['SAMPLE_ID']] = correct_data
        f1.close()
        for f2_line in f2:
            meta_data = json.loads(f2_line)
            res_dict[correct_data['SAMPLE_ID']] = meta_data
        f2.close()
        for d in data:
            idx = d['id'].decode()
            if idx in res_dict:
                f3.write(json.dumps(res_dict[idx]) + '\n')
            else:
                f3.write(json.dumps(backup_dict[idx]) + '\n')
                f4.write(idx+'\n')
        f3.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--master_addr', type=str, default='')
    parser.add_argument('--master_port', type=int, default=7878)
    args = parser.parse_args()

    input_path = "/nxchinamobile2/shared/img_datasets/laion115m"
    debug_path = "/nxchinamobile2/shared/jjh/laion115m-debug"
    output_path = "/nxchinamobile2/shared/jjh/laion115m_grounding"
    ids = []
    for dir in os.listdir(debug_path):
        ids.extend(os.listdir(os.path.join(input_path, dir)))
    ids = [name.split(".")[0] for name in ids if name.endswith(".tar")]
    ids.sort()
    select_ids = split_list_by_n(ids, args.world_size)[args.rank]
    process_list = []
    for idx in tqdm(select_ids):
        dir_name = "part-000"+idx[:2]
        output_dir_path = os.path.join(output_path, dir_name)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        meta_dir_path = os.path.join(debug_path, dir_name)
        tar_dir_path = os.path.join(input_path, dir_name)

        tar_filename = idx+".tar"
        tar_file_path = os.path.join(tar_dir_path, tar_filename)
        meta_filename = idx+".meta.jsonl"
        meta_file_path = os.path.join(meta_dir_path, meta_filename)
        backup_file_path = os.path.join(tar_dir_path, meta_filename)
        out_file_path = os.path.join(output_dir_path, meta_filename)
        correct_file_path = os.path.join(output_dir_path, "corrected_"+meta_filename)
        record_path = os.path.join(output_path, "record.txt")
        p = Process(target=combine_files, args=(tar_file_path, meta_file_path, correct_file_path, out_file_path, backup_file_path, record_path))
        p.start()
        process_list.append(p)
        if len(process_list) >= 36:
            for p in process_list:
                p.join()
            process_list = []
