import argparse
import json
import os
from tqdm import tqdm
from collections import defaultdict
import re
import webdataset as wds
from PIL import Image
import io
import math
import re
from visualize_grounding import vis_image

puncts = ['|', ':', ';', '@', '(', ')', '[', ']', '{', '}', '^', '\\', '/',
          '\'', '\"', '’', '`', '?', '$', '%', '#', '!', '&', '*', '+', ',', '.'
          ]


def findall_puncts(text):
    caption = text
    res = []
    for punct in puncts:
        beg = 0
        while text.find(punct, beg) != -1:
            pos = text.find(punct, beg)
            res.append(pos)
            beg = pos + 1
    for p in puncts:
        caption = caption.replace(p, '')
    for i, c in enumerate(caption):
        if c != ' ':
            break
        res.append(0)
    return res
    # return [sub.start() for sub in re.finditer(pattern, text)]


def process_dict(g_dict, add_map, caption):
    new_dict = defaultdict(dict)
    for obj in g_dict:
        for pos in g_dict[obj]:
            if int(pos) >= len(caption):
                continue
            new_pos = int(pos) + add_map[int(pos)]
            new_dict[obj][str(new_pos)] = g_dict[obj][pos]
    return new_dict


def remove_punctuation(text: str) -> str:
    punct = ['|', ':', ';', '@', '(', ')', '[', ']', '{', '}', '^', '\\', '/',
             '\'', '\"', '’', '`', '?', '$', '%', '#', '!', '&', '*', '+', ',', '.'
             ]
    for p in punct:
        text = text.replace(p, '')
    return text.lstrip()


from multiprocessing import Process


def revise_and_write(output_dir_path, file):
    with open(os.path.join(output_dir_path, file), "r", encoding='utf-8') as f1, open(
            os.path.join(output_dir_path, file + ".update"), "a", encoding='utf-8') as f2:
        for line in f1:
            pos_add_map = {}
            data = json.loads(line)
            if data['status'] != "success":
                continue
            caption = data['caption']
            # print("caption:", caption, 'len:', len(caption))

            punct_pos_list = findall_puncts(caption)

            caption_tmp = caption
            strip_caption = remove_punctuation(caption_tmp)
            try:
                assert len(strip_caption) == len(caption) - len(punct_pos_list)
            except Exception as e:
                print(caption)
                print(strip_caption)
                print(punct_pos_list)
            # print("caption:", caption, 'len:', len(caption))
            # print(punct_pos_list)
            cur_punct_num = 0
            for i in range(len(caption)):
                if i in punct_pos_list:
                    if i == 0:
                        pos_add_map[0] = 0 if 0 not in pos_add_map else pos_add_map[0] + 1
                    else:
                        pos_add_map[i - cur_punct_num] = cur_punct_num + 1
                    cur_punct_num += 1
            cur_punct_num = 0
            for i in range(len(caption)):
                if i in pos_add_map:
                    cur_punct_num = pos_add_map[i]
                    continue
                pos_add_map[i] = cur_punct_num
            # print(pos_add_map)
            try:
                data['groundings'] = process_dict(data['groundings'], pos_add_map, caption)
                data['original_groundings'] = process_dict(data['original_groundings'], pos_add_map, caption)
            except Exception as e:
                print("groundings:", data['groundings'])
                print("original_groundings:", data['original_groundings'])
                data['groundings'] = []
                data['original_groundings'] = []
                print(caption)
                print(strip_caption)
                print(punct_pos_list)
            f2.write(json.dumps(data, ensure_ascii=False) + '\n')
        f2.close()
        f1.close()


def compare_and_update(tar_id, tar_path, ouput_path):
    # with open(os.path.join(ouput_path, tar_id + ".meta.jsonl.update"), 'r', encoding='utf-8') as f1, open(
    #         os.path.join(ouput_path, tar_id + ".meta.jsonl.update2"), 'a', encoding='utf-8') as f2:
    with open(os.path.join(ouput_path, tar_id + ".meta.jsonl.update"), 'r', encoding='utf-8') as f1:
        tar_file_path = os.path.join(tar_path, tar_id + ".tar")
        dataset = wds.WebDataset(tar_file_path)
        ds_iter = iter(dataset)
        for line in f1:
            data = json.loads(line)
            if data['status'] != "success":
                continue
            ds = next(ds_iter)
            ds_caption = ds['txt'].decode()
            data['tar_caption'] = ds_caption
            assert(ds['id'].decode() == data['SAMPLE_ID'])
            try:
                assert (ds_caption == data['caption'])
            except Exception as e:
                print("tar_caption:", ds_caption)
                print("meta_caption:", data['caption'])

            # f2.write(json.dumps(data) + '\n')
        # f2.close()
        f1.close()


def is_small_object(json_object, thresh=0.1):
    for item, groundings in dict(json_object['task_data']['groundings']).items():
        for pos, boxes in groundings.items():
            for box in boxes:
                x_1, y_1, x_2, y_2 = box
                if (x_2 - x_1) * (y_2 - y_1) < thresh:
                    return True
    return False


def get_area_ratio(ref_coords):
    x_1, y_1, x_2, y_2 = ref_coords
    return (x_2 - x_1) * (y_2 - y_1)


def process_json(data, wds_data, thresh):
    for item, groundings in dict(data['task_data']['groundings']).items():
        for pos, boxes in groundings.items():
            for box in boxes:
                if get_area_ratio(box) < thresh:
                    img = wds_data['jpg']
                    pil_image = Image.open(io.BytesIO(img)).convert('RGB')
                    vis_image(pil_image, data, "./visual_output")
                    return


def pick_small_items(tar_id, part_id, thresh=0.1):
    dataset_path = "/nxchinamobile2/shared/img_datasets/filted_laion115m_grounding"
    part_path = os.path.join(dataset_path, "part-%05d" % part_id)
    tar_path = os.path.join(part_path, f"{tar_id}.tar")
    meta_path = os.path.join(part_path, f"{tar_id}.meta.jsonl")
    web = wds.WebDataset(tar_path)
    web_iter = iter(web)
    with open(meta_path, 'r') as f1:
        lines = f1.readlines()
        for line in tqdm(lines):
            data = json.loads(line)
            if data['task_data']['status'] != "success":
                continue
            wds_data = next(web_iter)
            assert wds_data['id'].decode() == data['task_data']['SAMPLE_ID']
            process_json(data, wds_data, thresh)


# if __name__ == '__main__':
#     grounding_path = '/nxchinamobile2/shared/img_datasets/clay1b/meta_jsonl_cleaned_data/meta_jsonl_all_laion2ben_merged2_translate'
#     for part in tqdm(os.listdir(grounding_path)):
#         part_path = os.path.join(grounding_path, part)
#         for file in os.listdir(part_path):
#             if file.endswith(".meta.jsonl"):
#                 file_path = os.path.join(part_path, file)
#                 try:
#                     with open(file_path, 'r', errors='ignore') as f1:
#                         for line in f1:
#                             try:
#                                 data = json.loads(line)
#                             except Exception:
#                                 print(file_path)
#                                 print(line)
#                 except Exception:
#                     print(file_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--master_addr', type=str, default='')
    parser.add_argument('--master_port', type=int, default=7878)
    args = parser.parse_args()

    grounding_path = '/nxchinamobile2/shared/img_datasets/laion115m_grounding_small_objects_optimized'
    tar_path = "/nxchinamobile2/shared/img_datasets/laion115m_grounding_small_objects_optimized"
    small_count = 0
    count = 0
    for part in tqdm(range(32, 47)):
        part_name = "part-%05d" % part
        part_path = os.path.join(grounding_path, part_name)
        for file in os.listdir(part_path):
            if not file.endswith('.meta.jsonl'):
                continue
            file_path = os.path.join(part_path, file)
            # print(file_path)
            tar_id = file.split('.')[0]
            tar_file = os.path.join(tar_path, f"part-000{part}", f"{tar_id}.tar")
            # wds_data = wds.WebDataset(tar_file)
            # wds_iter = iter(wds_data)
            contains_small = False
            with open(file_path, 'r', errors='ignore') as f1:
                for line in f1:
                    data = json.loads(line)
                    # if data['task_data']['status'] == 'success':
                    #     cur_wds = next(wds_iter)
                    if data['task_data']['small'] and not data['task_data']['bad']:
                        contains_small = True
                        break
                        # img = cur_wds['jpg']
                        # pil_image = Image.open(io.BytesIO(img)).convert('RGB')
                        # vis_image(pil_image, data, "./good_and_small")
                        small_count += 1
                    # if small_count == 50:
                    #     raise Exception("STOP")
                    count += 1
            if not contains_small:
                print(tar_file)
                print(file_path)
                # os.system(f"rm {tar_file}")
                # os.system(f"rm {file_path}")
        # print(small_count, count)
    # tar_path = "/nxchinamobile2/shared/img_datasets/filted_laion115m_grounding"
    #
    # tar_id = 3200005
    # part_id = 32
    #
    # tar_file = os.path.join(tar_path, f"part-000{part_id}", f"{tar_id}.tar")
    # meta_file = os.path.join(grounding_path, f"part-000{part_id}", f"{tar_id}.meta.jsonl")
    # wds_dataset = wds.WebDataset(tar_file)
    # wds_iter = iter(wds_dataset)
    # with open(meta_file, 'r', errors='ignore') as f1:
    #     for line in tqdm(f1):
    #         data = json.loads(line)
    #         if data['task_data']['status'] == 'success':
    #             wds_data = next(wds_iter)
    #             if data['task_data']['small']:
    #                 img = wds_data['jpg']
    #                 pil_image = Image.open(io.BytesIO(img)).convert('RGB')
    #                 vis_image(pil_image, data, "./3200005")


    # pick_small_items(tar_id, part_id, thresh=0.05)
# if __name__ == '__main__':
#     grounding_path = '/nxchinamobile2/shared/jjh/laion115m'
#     process_list = []
#     for part in os.listdir(grounding_path):
#         for file in os.listdir(os.path.join(grounding_path, part)):
#             ground_file = os.path.join(grounding_path, part, file_id+'.meta.jsonl')
#             p = Process(target=YOUR_FUNC,args=(ground_file,))
#             p.start()
#             process_list.append(p)
#     for p in process_list:
#         p.join()


# if __name__ == "__main__":
#     output_path = "/nxchinamobile2/shared/jjh/laion115m"
#     input_path = "/nxchinamobile2/shared/img_datasets/laion115m"
#     process_list = []
#     for dir_i, dir in enumerate(os.listdir(output_path)):
#         print("processing dir {} {}/{}...".format(os.listdir(output_path), dir_i, len(os.listdir(output_path))))
#         output_dir_path = os.path.join(output_path, dir)
#         files = os.listdir(output_dir_path)
#         tar_path = os.path.join(input_path, dir)
#         for file in tqdm(files):
#             tar_id = file.split('.')[0]
#             p = Process(target=compare_and_update, args=(tar_id, tar_path, output_dir_path))
#             p.start()
#             process_list.append(p)
#             if len(process_list) >= 400:
#                 for p in process_list:
#                     p.join()
#                 process_list = []

# def read_tar(tar_path):
#     return wds.WebDataset(tar_path)
#
# #
# # TOTAL_PIXEL = 345600
# # FACTOR_DICT = {}
# # mid = int(math.sqrt(TOTAL_PIXEL))
# # for i_t in range(mid+1)[1:]:
# #     if TOTAL_PIXEL % i_t == 0:
# #         FACTOR_DICT[i_t] = int(TOTAL_PIXEL / i_t)
# # print(FACTOR_DICT)
#
# SOLUTION = "480p"
# RESOLUTIONS = {"240p": (320, 240), "480p": (720, 480), "720p": (1280, 720), "1080p": (1920, 1080), "2K": (2560, 1440),
#                "4K": (4096, 2160)}
# TOTAL_PIXEL = RESOLUTIONS[SOLUTION][0] * RESOLUTIONS[SOLUTION][1]  # 480P resolution
# FACTOR_DICT = {}
# mid = int(math.sqrt(TOTAL_PIXEL))
# for i_t in range(mid + 1)[432:]:
#     if TOTAL_PIXEL % i_t == 0:
#         FACTOR_DICT[i_t] = int(TOTAL_PIXEL / i_t)
# vs = list(FACTOR_DICT.values())
# vs.reverse()
# ks = list(FACTOR_DICT.keys())
# ks.reverse()
# update_dict = dict(zip(vs, ks))
# FACTOR_DICT.update(update_dict)
# print(FACTOR_DICT)

# if __name__ == '__main__':

# text = "[ disc, entity:the reign of ragnarok [ disc 1 of 2 ]"
#
# def remove_puncs(caption):
#     r = "[+=^*<>{}「」【】()（）/\[\],.?，。？！:@¥%!@#$%&]"
#     caption = re.sub(r, ' ', caption)
#     return caption
#
# print(remove_puncs(text))
# width, height = [600, 800]
# loc = [100, 100]
# loc = [l / width if i % 2 == 0 else l / height for i, l in enumerate(loc)]
# print(loc)
#
# if __name__ == "__main__":
#     # nlp = spacy.load("en_core_web_trf")
#     input_path = "/gpfs/gpfs1/zphz/img_datasets/laion115m/part-00032"
#     output_path = "/gpfs/gpfs1/zphz/jjh/test_dataset/part-00032"
#     ids = [0, 1, 2, 3, 4, 5]
#     for idx in ids:
#         res = {}
#         tar_filename = "{}.tar".format(3200000+idx)
#         tar_dataset = read_tar(os.path.join(input_path, tar_filename))
#         meta_filename = "{}.meta.jsonl".format(3200000+idx)
#         with open(os.path.join(input_path, meta_filename), 'r', encoding='utf-8') as f1, open(os.path.join(output_path, meta_filename), 'a', encoding='utf-8') as f2:
#             for data, line in zip(tar_dataset, f1):
#                 meta_data = json.loads(line)
#                 if meta_data['status'] == "success":
#                     size = (int(meta_data['width']), int(meta_data['height']))
#                     index = data['id'].decode()
#                     sample_id = meta_data['SAMPLE_ID']
#                     assert (str(index) == str(sample_id))
#                     image_b = data['jpg']
#                     image = Image.open(io.BytesIO(image_b)).convert('RGB')
#                     image.save("test.png", format="png")
#                     caption = data['txt']
#
#                     ret = parse_and_grounding_multi_class(image, caption, str(idx), nlp, output_path, True)
#                     meta_data.update(ret)
#                 f2.write(json.dumps(meta_data, ensure_ascii=False) + '\n')
#                 break
#         f1.close()
#         f2.close()
#         break
#     print("done")
