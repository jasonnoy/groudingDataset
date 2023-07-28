import json
import os
from tqdm import tqdm
import re
import webdataset as wds
from PIL import Image
import io
import math
import re

puncts = ["|", ":", ";", "@", "\(", "\)", "\[", "\]", "\{", "\}", "^", '\\\\', "/",
          '"', "’", "`", "\?", "$", "%", "#", "!", "&", "\*", "\+", ",", "."
          ]


def findall(text, pattern):
    return [sub.start() for sub in re.finditer(pattern, text)]


def process_dict(g_dict, add_map):
    print(g_dict)
    new_dict = {}
    for obj in g_dict:
        for pos in g_dict[obj]:
            new_pos = pos + add_map[pos]
            new_dict[obj][new_pos] = g_dict[obj][pos]
    return new_dict


if __name__ == "__main__":
    output_path = "/nxchinamobile2/shared/jjh/laion115m"
    for dir_i, dir in enumerate(os.listdir(output_path)):
        print("processing dir {}/{}...".format(dir_i, len(os.listdir(output_path))))
        output_dir_path = os.path.join(output_path, dir)
        files = os.listdir(output_dir_path)
        for file in tqdm(files):
            with open(os.path.join(output_dir_path, file), "r", encoding='utf-8') as f1, open(
                    os.path.join(output_dir_path, file + ".update"), "a", encoding='utf-8') as f2:
                for line in f1:
                    punct_pos_list = []
                    pos_add_map = {}
                    data = json.loads(line)
                    if data['status'] != "success":
                        continue
                    caption = data['caption']
                    for punct in puncts:
                        # print(punct)
                        punct_pos_list.extend(findall(caption, punct))
                    cur_punct_num = 0
                    for i in range(len(caption)):
                        if i in punct_pos_list:
                            pos_add_map[i - cur_punct_num] = cur_punct_num + 1
                            cur_punct_num += 1
                    data['groundings'] = process_dict(data['groundings'], pos_add_map)
                    data['original_groundings'] = process_dict(data['original_groundings'], pos_add_map)
                    f2.write(json.dumps(data, ensure_ascii=False) + '\n')
                f2.close()
                f1.close()

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
