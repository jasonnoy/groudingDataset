import os
import io
import numpy as np
from PIL import Image
import webdataset as wds
import torch.utils.data as data
import re
import math
import torch
from GLIP.maskrcnn_benchmark.structures.image_list import to_image_list


SOLUTION = "480p"
RESOLUTIONS = {"240p": [320, 240], "480p": [720, 480], "720p": [1280, 720], "1080p": [1920, 1080], "2K": [2560, 1440], "4K": [4096, 2160]}
TOTAL_PIXEL = RESOLUTIONS[SOLUTION][0] * RESOLUTIONS[SOLUTION][1]  # 480P resolution
FACTOR_DICT = {}
mid = int(math.sqrt(TOTAL_PIXEL))
for i_t in range(mid+1)[1:]:
    if TOTAL_PIXEL % i_t == 0:
        FACTOR_DICT[i_t] = int(TOTAL_PIXEL / i_t)
for k, v in FACTOR_DICT.items():
    FACTOR_DICT[v] = k


def pil_loader(image_b):
    pil_image = Image.open(io.BytesIO(image_b)).convert('RGB')
    # pil_image = pil_image.resize((800))
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


def create_positive_map(tokenized, tokens_positive):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            try:
                beg_pos = tokenized.char_to_token(beg)
                end_pos = tokenized.char_to_token(end - 1)
            except Exception as e:
                print("beg:", beg, "end:", end)
                print("token_positive:", tokens_positive)
                # print("beg_pos:", beg_pos, "end_pos:", end_pos)
                raise e
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos: end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


def create_positive_map_label_to_token_from_positive_map(positive_map, plus=0):
    positive_map_label_to_token = {}
    for i in range(len(positive_map)):
        positive_map_label_to_token[i + plus] = torch.nonzero(positive_map[i], as_tuple=True)[0].tolist()
    return positive_map_label_to_token


def compute_image_shape(original_shape):
    ratio = original_shape[0] / original_shape[1]
    edge = int(math.sqrt(TOTAL_PIXEL/ratio))
    if edge in FACTOR_DICT:
        return [FACTOR_DICT[edge], edge]
    prev = 1
    for cur in FACTOR_DICT.keys():
        if cur > edge > prev:
            return [FACTOR_DICT[cur], cur]
        prev = cur
    return RESOLUTIONS[SOLUTION]  # just in case


class Laion(data.Dataset):
    """ Laion dataset.

    Args:
        root (string): part directory where tar and meta files are at.
        index (string): index to tar and meta file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
    """

    def __init__(self, index, root, nlp, tokenizer, transforms=None, rpn_architecture="VLDYHEAD", batch_size=100):
        self.tokenizer = tokenizer
        self.root = root
        self.transform = transforms
        self.nlp = nlp
        self.rpn_architecture = rpn_architecture

        wds_ds = wds.WebDataset(os.path.join(root, "{}.tar".format(index)))
        self.samples = [[d['id'].decode(), pil_loader(d['jpg']), d['txt'].decode()] for d in wds_ds]
        # if self.transform is None:
        #     images = [s[1] for s in self.samples]
        # else:
        #     images = [self.transform(s[1]) for s in self.samples]
        # tensors = []
        # print("Processing dataset...")
        # for i in tqdm(range(0, len(self.samples), batch_size)):
        #     batch_images = images[i: i+batch_size]
        #     tensors.extend(to_image_list(batch_images, size_divisible=32).tensors)
        # assert (len(tensors) == len(self.samples))
        # for i in range(len(tensors)):
        #     self.samples[i][1] = tensors[i]

    def __getitem__(self, index):
        idx, image, caption = self.samples[index]
        image_size = image.shape[-2:]
        image_resize_shape = compute_image_shape(image_size)

        if self.transform is not None:
            image = self.transform(image, image_resize_shape)
        doc = self.nlp(caption)
        nouns = [t.text.lower() for t in doc.noun_chunks]
        empty_nouns = False
        if len(nouns) == 0:
            print("No entities found, using caption as entity, caption: {}".format(caption))
            nouns = [caption.lower()]
            empty_nouns = True
        entity_dict = {}
        new_entities = []
        # to handle duplicates in entities
        for chunk in nouns:
            if chunk not in entity_dict:
                new_entities.append(chunk)
                entity_dict[chunk] = 0
            else:
                entity_dict[chunk] += 1
                new_entities.append("{}-{}".format(chunk, entity_dict[chunk]))
        new_to_old_entity = dict(zip(new_entities, nouns))
        if not empty_nouns:
            new_entity_to_id = dict(zip(new_entities, [noun_chunk[0].idx for noun_chunk in
                                                       doc.noun_chunks]))  # starting position of the first token
        else:
            # use caption as only entity
            new_entity_to_id = {new_entities[0]: 0}

        tokenized = self.tokenizer([caption], return_tensors="pt")
        tokens_positive = []
        for entity in nouns:
            # want no overlays
            found = {(0, 0)}
            for m in re.finditer(entity, caption.lower()):
                if (m.start(), m.end()) not in found:
                    tokens_positive.append([[m.start(), m.end()]])
                    found.add((m.start(), m.end()))

        # process positive map
        positive_map = create_positive_map(tokenized, tokens_positive)

        if self.rpn_architecture == "VLDYHEAD":
            plus = 1
        else:
            plus = 0
        # positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map, plus=plus)
        return image, new_entities, positive_map, new_to_old_entity, new_entity_to_id

    def __len__(self):
        return len(self.samples)
