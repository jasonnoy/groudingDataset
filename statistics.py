import json
import multiprocessing
import os
from tqdm import tqdm


def count_part(part_path, q):
    total = 0
    total_pics = 0
    valid_num = 0
    num_chunks = 0
    total_str_len = 0
    num_bounds = 0
    total_width = 0
    total_height = 0
    for file in tqdm(os.listdir(part_path)):
        if not file.endswith(".jsonl"):
            continue
        file_path = os.path.join(part_path, file)
        with open(file_path, "r") as f:
            for l in f:
                total += 1
                data = json.loads(l)
                d = data['task_data']
                if d['status'] != 'success':
                    continue
                total_pics += 1
                if d['bad']:
                    continue
                total_width += d['width']
                total_height += d['height']
                valid_num += 1
                total_str_len += len(d['caption'])
                num_chunks += len(d['groundings'].keys())
                for k, v in d['groundings'].items():
                    for pos, coords in v.items():
                        num_bounds += len(coords)
    q.put((total, total_pics, valid_num, num_chunks, total_str_len, num_bounds, total_width, total_height))
    

if __name__ == "__main__":
    pool_size = 16
    total = 0
    total_pics = 0
    valid_num = 0
    num_chunks = 0
    total_str_len = 0
    num_bounds = 0
    total_width = 0
    total_height = 0

    q = multiprocessing.Queue()
    pool = []

    dataset_path = "/share/img_datasets/laion115m_grounding_small_objects_optimized"
    for part in os.listdir(dataset_path):
        part_path = os.path.join(dataset_path, part)
        p = multiprocessing.Process(target=count_part, args=(part_path, q))
        pool.append(p)
        p.start()

    for p in pool:
        p.join()

    for _ in pool:
        p_total, p_total_pics, p_valid_num, p_num_chunks, p_total_str_len, p_num_bounds, p_total_width, p_total_height = q.get()
        total += p_total
        total_pics += p_total_pics
        valid_num += p_valid_num
        num_chunks += p_num_chunks
        total_str_len += p_total_str_len
        num_bounds += p_num_bounds
        total_width += p_total_width
        total_height += p_total_height

    print("total:", total)
    print("total_pics:", total_pics)
    print("valid_num:", valid_num)
    print("avg_width:", total_width / valid_num)
    print("avg_height:", total_height / valid_num)
    print("total_str_len:", total_str_len)
    print("avg_str_len:", total_str_len / valid_num)
    print("num_chunks:", num_chunks)
    print("num_bounds:", num_bounds)
    print("avg_bounds per chunk:", num_bounds / num_chunks)
