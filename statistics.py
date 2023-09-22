import json
import os


total = 0
total_pics = 0
valid_num = 0
num_chunks = 0
total_str_len = 0
num_bounds = 0
total_width = 0
total_height = 0

dataset_path = "/share/img_datasets/laion115m_grounding_small_objects_optimized"
for part in os.listdir(dataset_path):
    part_path = os.path.join(dataset_path, "part-%05d" % part)
    for file in os.listdir(part_path):
        if not file.endswith(".json"):
            continue
        file_path = os.path.join(part_path, file)
        with open(file_path, "r") as f:
            for l in f:
                total += 1
                data = json.loads(l)
                d = data['task_data']
                if not d['success']:
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
                    for pos, coords in v:
                        num_bounds += len(coords)

print("total:", total)
print("total_pics:", total_pics)
print("valid_num:", valid_num)
print("avg_width:", total_width / total_pics)
print("avg_height:", total_height / total_pics)
print("total_str_len:", total_str_len)
print("avg_str_len:", total_str_len / valid_num)
print("num_chunks:", num_chunks)
print("num_bounds:", num_bounds)
print("avg_bounds per chunk:", num_bounds / num_chunks)
