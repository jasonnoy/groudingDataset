import cv2
import torch
import re
import numpy as np
from typing import List, Union
import nltk
import inflect
from transformers import AutoTokenizer
from torchvision import transforms as T
import pdb
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.dist import get_iou
from maskrcnn_benchmark.structures.image_list import to_image_list, ImageList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.utils import cv2_util

engine = inflect.engine()

import timeit

class to_bgr_transform(object):
    def __init__(self, to_bgr255):
        self.to_bgr255 = to_bgr255

    def __call__(self, image):
        if self.to_bgr255:
            image = image * 255
        else:
            image = image[[2, 1, 0]]
        return image


class GLIPDemo(object):
    def __init__(self,
                 cfg,
                 confidence_threshold=0.7,
                 min_image_size=None,
                 show_mask_heatmaps=False,
                 masks_per_dim=5,
                 load_model=True,
                 nlp=None
                 ):
        self.cfg = cfg.clone()
        if load_model:
            self.model = build_detection_model(cfg)
            self.model.eval()
            self.device = torch.device(cfg.MODEL.DEVICE)
            self.model.to(self.device)
        self.min_image_size = min_image_size
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim
        self.nlp = nlp

        save_dir = cfg.OUTPUT_DIR
        if load_model:
            checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
            _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        # used to make colors for each tokens
        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold

        self.tokenizer = self.build_tokenizer()

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.ToTensor(),
                to_bgr_transform(cfg.INPUT.TO_BGR255),
                normalize_transform,
            ]
        )
        return transform

    def build_tokenizer(self):
        cfg = self.cfg
        tokenizer = None
        if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "bert-base-uncased":
            tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.LOCAL_PATH)
        elif cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
            from transformers import CLIPTokenizerFast
            if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
                tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                              from_slow=True, mask_token='ðŁĴĳ</w>')
            else:
                tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                              from_slow=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.LOCAL_PATH)
        return tokenizer

    def run_ner(self, caption):
        noun_phrases = find_noun_phrases(caption)
        noun_phrases = [remove_punctuation(phrase) for phrase in noun_phrases]
        noun_phrases = [phrase for phrase in noun_phrases if phrase != '']
        relevant_phrases = noun_phrases
        labels = noun_phrases
        self.entities = labels

        tokens_positive = []

        for entity, label in zip(relevant_phrases, labels):
            try:
                # search all occurrences and mark them as different entities
                for m in re.finditer(entity, caption.lower()):
                    tokens_positive.append([[m.start(), m.end()]])
            except:
                print("noun entities:", noun_phrases)
                print("entity:", entity)
                print("caption:", caption.lower())

        return tokens_positive

    def inference(self, original_image, original_caption):
        predictions = self.compute_prediction(original_image, original_caption)
        top_predictions = self._post_process_fixed_thresh(predictions)
        return top_predictions

    def run_on_web_image(self,
                         original_image,
                         original_caption,
                         thresh=0.5,
                         custom_entity=None,
                         alpha=0.0,
                         color=255):
        self.color = color
        predictions = self.compute_prediction(original_image, original_caption, custom_entity)
        top_predictions = self._post_process(predictions, thresh)
        result = original_image.copy()
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)
        result = self.overlay_boxes(result, top_predictions)
        result = self.overlay_entity_names(result, top_predictions)
        if self.cfg.MODEL.MASK_ON:
            result = self.overlay_mask(result, top_predictions)
        return result, top_predictions

    def run_on_image(self,
                     original_image,
                     original_caption,
                     thresh=0.5,
                     custom_entity=None,
                     alpha=0.0,
                     save_img=False):
        predictions = self.compute_prediction(original_image, original_caption, custom_entity)
        top_predictions = self._post_process(predictions, thresh)
        result = None
        if save_img:
            result = original_image.copy()
            result = self.overlay_boxes(result, top_predictions)
        return result, top_predictions

    def run_on_batched_images(self,
                              images,
                              image_sizes,
                              captions,
                              positive_map_label_to_tokens,
                              entity_lists,
                              origin_images=None,
                              thresh=0.5,
                              save_img=False):
        images = ImageList(images, image_sizes)
        images = images.to(self.device)
        # print("captions:", captions)
        # print("positive_map_label_to_tokens:", positive_map_label_to_tokens)
        with torch.no_grad():
            predictions = self.model(images, captions, positive_map_label_to_tokens)
            predictions = [p.to("cpu") for p in predictions]
        # for origin_image, prediction in zip(origin_images, predictions):
        #     height, width = origin_image.shape[:-1]
        #     prediction = prediction.resize((width, height))
        predictions = [prediction.resize((origin_image.shape[1], origin_image.shape[0])) for prediction, origin_image in
                       zip(predictions, origin_images)]
        list_locations = get_entity_list_locs(entity_lists)
        top_predictions = [self._post_process(prediction, list_loc, thresh) for
                           prediction, entity_list, list_loc in zip(predictions, entity_lists, list_locations)]
        results = None
        if save_img:
            results = [image.copy() for image in origin_images]
            results = [self.overlay_boxes(result, top_prediction) for result, top_prediction in
                       zip(results, top_predictions)]
        images.to("cpu")
        del images
        torch.cuda.empty_cache()
        return results, top_predictions

    def visualize_with_predictions(self,
                                   original_image,
                                   predictions,
                                   thresh=0.5,
                                   alpha=0.0,
                                   box_pixel=3,
                                   text_size=1,
                                   text_pixel=2,
                                   text_offset=10,
                                   text_offset_original=4):
        height, width = original_image.shape[:-1]
        predictions = predictions.resize((width, height))
        top_predictions = self._post_process(predictions, thresh)

        result = original_image.copy()
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)
        result = self.overlay_boxes(result, top_predictions, alpha=alpha, box_pixel=box_pixel)
        result = self.overlay_entity_names(result, top_predictions, text_size=text_size, text_pixel=text_pixel,
                                           text_offset=text_offset, text_offset_original=text_offset_original)
        if self.cfg.MODEL.MASK_ON:
            result = self.overlay_mask(result, top_predictions)
        return result, top_predictions

    def _post_process_fixed_thresh(self, predictions):
        scores = predictions.get_field("scores")
        labels = predictions.get_field("labels").tolist()
        thresh = scores.clone()
        for i, lb in enumerate(labels):
            if isinstance(self.confidence_threshold, float):
                thresh[i] = self.confidence_threshold
            elif len(self.confidence_threshold) == 1:
                thresh[i] = self.confidence_threshold[0]
            else:
                thresh[i] = self.confidence_threshold[lb - 1]
        keep = torch.nonzero(scores > thresh).squeeze(1)
        predictions = predictions[keep]

        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def filter_iou(self, prediction, threshold=0.95):
        # predictions in descending order
        qualified = {}
        labels = prediction.get_field("labels").tolist()
        scores = prediction.get_field("scores").tolist()
        for idx, bbox in enumerate(prediction.bbox):
            top_left, bottom_right = bbox[:2].tolist(), bbox[2:].tolist()
            top_left = (top_left[0], top_left[1])
            bottom_right = (bottom_right[0], bottom_right[1])
            if len(qualified) == 0:
                qualified[(top_left, bottom_right)] = idx
                continue
            add = True
            for tl, rb in *qualified.keys(),:
                iou = get_iou(top_left, bottom_right, tl, rb)
                if get_iou(top_left, bottom_right, tl, rb) > threshold:
                    add = False
                    break
            if add:
                qualified[(top_left, bottom_right)] = idx
        ids = list(qualified.values())
        return prediction[ids]

    def compute_prediction(self, original_image, original_caption, custom_entities=None):
        # image
        image = self.transforms(original_image)
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        if custom_entities:
            self.entities = custom_entities
            tokenized = self.tokenizer([original_caption], return_tensors="pt")
            tokens_positive = []
            for entity in custom_entities:
                try:
                    # want no overlays
                    found = {(0, 0)}
                    for m in re.finditer(entity, original_caption.lower()):
                        if (m.start(), m.end()) not in found:
                            tokens_positive.append([[m.start(), m.end()]])
                            found.add((m.start(), m.end()))
                except:
                    print("noun entities:", custom_entities)
                    print("entity:", entity)
                    print("caption:", original_caption.lower())
        else:
            # caption
            print("Empty entities")
            if isinstance(original_caption, list):
                # we directly provided a list of category names
                self.entities = original_caption
                caption_string = ""
                tokens_positive = []
                seperation_tokens = " . "
                for word in original_caption:
                    tokens_positive.append([len(caption_string), len(caption_string) + len(word)])
                    caption_string += word
                    caption_string += seperation_tokens
                original_caption = caption_string
                tokenized = self.tokenizer([caption_string], return_tensors="pt")
                tokens_positive = [tokens_positive]
            else:
                tokenized = self.tokenizer([original_caption], return_tensors="pt")
                tokens_positive = self.run_ner(original_caption)
        # process positive map
        positive_map = create_positive_map(tokenized, tokens_positive)

        if self.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
            plus = 1
        else:
            plus = 0

        positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map, plus=plus)
        self.plus = plus
        self.positive_map_label_to_token = positive_map_label_to_token
        # positive_maps.append(positive_map_label_to_token)
        # tic = timeit.time.perf_counter()

        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list, captions=[original_caption], positive_map=positive_map_label_to_token)
            predictions = [o.to(self.cpu_device) for o in predictions]
        # print("inference time per image: {}".format(timeit.time.perf_counter() - tic))

        # # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        # if predictions[i].has_field("mask"):
        #     # if we have masks, paste the masks in the right position
        #     # in the image, as defined by the bounding boxes
        #     masks = predictions[i].get_field("mask")
        #     # always single image is passed at a time
        #     masks = self.masker([masks], [predictions[i]])[0]
        #     predictions[i].add_field("mask", masks)

        return prediction

    def filter_object(self, prediction, list_loc):
        ids = []
        labels = prediction.get_field("labels").tolist()
        for idx, l in enumerate(labels):
            if l > list_loc[1] or l <= list_loc[0]:
                continue
            else:
                ids.append(idx)
        return prediction[ids]

    def _post_process(self, prediction, list_loc, threshold=0.5):
        scores = prediction.get_field("scores")
        # print("before post process")
        # print("scores:", scores)
        labels = prediction.get_field("labels").tolist()
        # print("labels:", labels)
        # print("scores:", scores)
        thresh = scores.clone()
        for i, lb in enumerate(labels):
            if isinstance(self.confidence_threshold, float):
                thresh[i] = threshold
            elif len(self.confidence_threshold) == 1:
                thresh[i] = threshold
            else:
                thresh[i] = self.confidence_threshold[lb - 1]
        keep = torch.nonzero(scores > thresh).squeeze(1)
        prediction = prediction[keep]
        scores = prediction.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        prediction = prediction[idx]
        # print("after score filter:")
        # print("scores:", prediction.get_field("scores"))
        # print("labels:", prediction.get_field("labels"))
        prediction = self.filter_object(prediction, list_loc)
        # print("after object filter:")
        # print("scores:", prediction.get_field("scores"))
        # print("labels:", prediction.get_field("labels"))
        prediction = self.filter_iou(prediction)
        # print("final:")
        # print("scores:", prediction.get_field("scores"))
        # print("labels:", prediction.get_field("labels"))
        return prediction

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = (30 * (labels[:, None] - 1) + 1) * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        try:
            colors = (colors * 0 + self.color).astype("uint8")
        except:
            pass
        return colors

    def overlay_boxes(self, image, predictions, alpha=0.5, box_pixel=3):
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()
        new_image = image.copy()
        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            new_image = cv2.rectangle(
                new_image, tuple(top_left), tuple(bottom_right), tuple(color), box_pixel)

        # Following line overlays transparent rectangle over the image
        image = cv2.addWeighted(new_image, alpha, image, 1 - alpha, 0)

        return image

    def overlay_scores(self, image, predictions):
        scores = predictions.get_field("scores")
        boxes = predictions.bbox

        for box, score in zip(boxes, scores):
            box = box.to(torch.int64)
            image = cv2.putText(image, '%.3f' % score,
                                (int(box[0]), int((box[1] + box[3]) / 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        return image

    def overlay_entity_names(self, image, predictions, entities, text_size=1.0, text_pixel=2, text_offset=10,
                             text_offset_original=4, custom_labels=None):
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        colors = self.compute_colors_for_labels(predictions.get_field("labels")).tolist()
        if custom_labels:
            new_labels = custom_labels
        else:
            new_labels = []
            for i in labels:
                if i <= len(entities):
                    new_labels.append(entities[i])
                else:
                    new_labels.append('object')
        boxes = predictions.bbox

        template = "{}:{:.2f}"
        previous_locations = []
        for box, score, label, color in zip(boxes, scores, new_labels, colors):
            x, y = box[:2]
            s = template.format(label, score).replace("_", " ").replace("(", "").replace(")", "")
            for x_prev, y_prev in previous_locations:
                if abs(x - x_prev) < abs(text_offset) and abs(y - y_prev) < abs(text_offset):
                    y -= text_offset
            cv2.putText(
                image, s, (int(x), int(y) - text_offset_original), cv2.FONT_HERSHEY_SIMPLEX, text_size, color,
                text_pixel, cv2.LINE_AA
            )
            previous_locations.append((int(x), int(y)))
        return image

    def overlay_mask(self, image, predictions):
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        colors = self.compute_colors_for_labels(labels).tolist()

        # import pdb
        # pdb.set_trace()
        # masks = masks > 0.1

        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None].astype(np.uint8)
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(image, contours, -1, color, 2)

        composite = image

        return composite

    def create_mask_montage(self, image, predictions):
        masks = predictions.get_field("mask")
        masks_per_dim = self.masks_per_dim
        masks = L.interpolate(
            masks.float(), scale_factor=1 / masks_per_dim
        ).byte()
        height, width = masks.shape[-2:]
        max_masks = masks_per_dim ** 2
        masks = masks[:max_masks]
        # handle case where we have less detections than max_masks
        if len(masks) < max_masks:
            masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
            masks_padded[: len(masks)] = masks
            masks = masks_padded
        masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
        result = torch.zeros(
            (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
        )
        for y in range(masks_per_dim):
            start_y = y * height
            end_y = (y + 1) * height
            for x in range(masks_per_dim):
                start_x = x * width
                end_x = (x + 1) * width
                result[start_y:end_y, start_x:end_x] = masks[y, x]

        return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET), None


def create_positive_map_label_to_token_from_positive_map(positive_map, plus=0):
    positive_map_label_to_token = {}
    for i in range(len(positive_map)):
        positive_map_label_to_token[i + plus] = torch.nonzero(positive_map[i], as_tuple=True)[0].tolist()
    return positive_map_label_to_token


def create_positive_map(tokenized, tokens_positive):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            try:
                beg_pos = tokenized.char_to_token(beg)
                end_pos = tokenized.char_to_token(end - 1)
            except Exception as e:
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


def find_noun_phrases(caption: str) -> List[str]:
    caption = caption.lower()
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)

    grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(pos_tags)

    noun_phrases = list()
    for subtree in result.subtrees():
        if subtree.label() == 'NP':
            noun_phrases.append(' '.join(t[0] for t in subtree.leaves()))

    return noun_phrases


def remove_punctuation(text: str) -> str:
    punct = ['|', ':', ';', '@', '(', ')', '[', ']', '{', '}', '^',
             '\'', '\"', '’', '`', '?', '$', '%', '#', '!', '&', '*', '+', ',', '.'
             ]
    for p in punct:
        text = text.replace(p, '')
    return text.strip()


def get_entity_list_locs(entity_lists):
    cur_beg = 0
    res = []
    for entity_list in entity_lists:
        beg = cur_beg
        end = cur_beg + len(entity_list)
        res.append((beg, end))
        cur_beg = end
    return res
