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

SOLUTION = "720p"
RESOLUTIONS = {"240p": (320, 240), "480p": (720, 480), "720p": (1280, 720), "1080p": (1920, 1080), "2K": (2560, 1440),
               "4K": (4096, 2160)}
TOTAL_PIXEL = RESOLUTIONS[SOLUTION][0] * RESOLUTIONS[SOLUTION][1]  # 480P resolution
FACTOR_DICT = {}
mid = int(math.sqrt(TOTAL_PIXEL))
for i_t in range(mid + 1)[1:]:
    if TOTAL_PIXEL % i_t == 0:
        FACTOR_DICT[i_t] = int(TOTAL_PIXEL / i_t)
vs = list(FACTOR_DICT.values())
vs.reverse()
ks = list(FACTOR_DICT.keys())
ks.reverse()
update_dict = dict(zip(vs, ks))
FACTOR_DICT.update(update_dict)


def pil_loader(image_b):
    pil_image = Image.open(io.BytesIO(image_b)).convert('RGB')
    # pil_image = pil_image.resize((800))
    # convert to BGR format
    return pil_image


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
    ratio = original_shape[1] / original_shape[0]
    edge = int(math.sqrt(TOTAL_PIXEL / ratio))
    if edge in FACTOR_DICT:
        return edge, FACTOR_DICT[edge]
    prev = 1
    for cur in FACTOR_DICT.keys():
        if edge > cur:
            prev = cur
            continue
        fit = prev if edge - prev < cur - edge else cur
        return fit, FACTOR_DICT[fit]
    return RESOLUTIONS[SOLUTION]  # just in case


class Laion(data.Dataset):
    """ Laion dataset.

    Args:
        root (string): part directory where tar and meta files are at.
        index (string): index to tar and meta file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
    """

    def __init__(self, index, root, nlp, tokenizer, transforms=None, rpn_architecture="VLDYHEAD", size_divisible=32):
        self.tokenizer = tokenizer
        self.root = root
        self.transform = transforms
        self.nlp = nlp
        self.rpn_architecture = rpn_architecture
        self.size_divisible = size_divisible

        wds_ds = wds.WebDataset(os.path.join(root, "{}.tar".format(index)))
        self.ids = [d['id'].decode() for d in wds_ds]
        self.captions = [d['txt'].decode() for d in wds_ds]

        images = [pil_loader(d['jpg']) for d in wds_ds]
        self.original_images = [np.array(image)[:, :, [2, 1, 0]] for image in images]
        images = [self.preprocess_image(img) for img in images]

        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        print("max size:", max_size)
        if self.size_divisible > 0:
            import math
            stride = self.size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(images),) + max_size
        batched_imgs = images[0].new(*batch_shape).zero_()
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        self.images = batched_imgs
        self.image_sizes = [im.shape[-2:] for im in images]

    def __getitem__(self, index):
        idx = self.ids[index]
        image = self.images[index]
        caption = self.captions[index]
        r = "[+=^*<>{}「」【】()（）/\[\]]"
        caption = re.sub(r, ' ', caption)
        origin_image = np.array(image)[:, :, [2, 1, 0]]

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
            try:
                for m in re.finditer(entity, caption.lower()):
                    if (m.start(), m.end()) not in found:
                        tokens_positive.append([[m.start(), m.end()]])
                        found.add((m.start(), m.end()))
            except Exception as e:
                raise ValueError("caption:{}, entity:{}".format(entity, caption.lower()))

        # process positive map
        positive_map = create_positive_map(tokenized, tokens_positive)

        return image, caption, positive_map, new_entities, new_to_old_entity, new_entity_to_id, origin_image, idx

    def preprocess_image(self, image):
        image_shape = image.size
        image_resize_shape = compute_image_shape(image_shape)
        image = image.resize(image_resize_shape)
        image = np.array(image)[:, :, [2, 1, 0]]

        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.samples)
