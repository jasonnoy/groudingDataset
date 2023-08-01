from multiprocessing import Process
import argparse
import json
import os
import sys
from tqdm import tqdm
import webdataset
import warnings
import math
warnings.filterwarnings("ignore")


def split_list_by_n(origin_list, n):
    step = math.ceil(len(origin_list) / n)
    res = []
    for i in range(0, len(origin_list), step):
        res.append(origin_list[i:i + step])
    return res


def write_dataset(meta_path, tar_path, out_path):
    tar_dataset = webdataset.WebDataset(tar_path)
    ids = []
    with open(meta_path, 'r', encoding='utf-8') as f1:
        for line in f1:
            data = json.loads(line)
            ids.append(data['SAMPLE_ID'])
    f1.close()
    print("ids:", ids)

    sink = webdataset.TarWriter(out_path, encoder=False)
    for line in tar_dataset:
        if line['id'].decode() in ids:
            print("write:", {"id": line['id'], "txt": line['txt'], "jpg": line['jpg']})
            sink.write({"id": line['id'], "txt": line['txt'], "jpg": line['jpg']})
    sink.close()


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--master_addr', type=str, default='')
    parser.add_argument('--master_port', type=int, default=7878)
    args = parser.parse_args()

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    rank = args.rank
    local_rank = args.local_rank
    world_size = args.world_size

    process_list = []

    input_path = "/nxchinamobile2/shared/img_datasets/laion115m"
    output_path = "/nxchinamobile2/shared/jjh/laion115m-debug"
    for dir_i, dir_name in enumerate(os.listdir(output_path)):
        print("processing dir {} {}/{}...".format(dir_name, dir_i, len(os.listdir(output_path))))
        output_dir_path = os.path.join(output_path, dir_name)
        input_dir_path = os.path.join(input_path, dir_name)
        debug_files = os.listdir(output_dir_path)
        debug_files = [filename for filename in debug_files if "error_" in filename]
        for filename in tqdm(debug_files):
            idx = filename[6:13]
            input_tar_path = os.path.join(input_dir_path, f"{idx}.tar")
            output_tar_path = os.path.join(output_dir_path, f"{idx}.tar")
            meta_path = os.path.join(output_dir_path, filename)
            p = Process(target=write_dataset, args=(meta_path, input_tar_path, output_tar_path))
            p.start()
            process_list.append(p)
            if len(process_list) >= 1:
                for p in process_list:
                    p.join()
                process_list = []
