import torch.utils.data
from GLIP.maskrcnn_benchmark.data.collate_batch import BatchGroundingCollator
from tqdm import tqdm
from functools import reduce
from operator import add
from .utils import *


class BaseGrounder:
    def __init__(self, data_collator, grounder_model, batch_size):
        self.data_collator = data_collator
        self.grounder_model = grounder_model
        self.batch_size = batch_size

    def __call__(self, dataset, batch_size, output_path, *args, save_img=False, num_workers=4, **kwargs):
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=num_workers,
                                                 batch_size=self.batch_size, collate_fn=self.data_collator)
        if save_img:
            print("Saving images to {}".format(output_path))

        return self.get_groundings(dataloader, output_path, *args, save_img=save_img, **kwargs)

    def get_groundings(self, dataloader, output_path, save_img=False, output_decorator=None, thresh=0.55):
        total_groundings = []
        for i, batch in tqdm(enumerate(dataloader)):
            origin_images = batch[7]
            results, preds = self.grounder_model(*batch[:5], origin_images=origin_images, thresh=thresh, save_img=save_img)
            new_entities = batch[4]
            entire_entities = reduce(add, new_entities)
            new_to_old_entities = batch[5]
            new_entity_to_ids = batch[6]
            image_ids = batch[8]
            captions = batch[9]
            if results:
                for result, pred, caption, new_entity_to_id, new_to_old_entity, index in zip(results, preds, captions,
                                                                                             new_entity_to_ids,
                                                                                             new_to_old_entities,
                                                                                             image_ids):
                    new_labels = get_label_names(pred, self.grounder_model, entire_entities)
                    old_labels = [new_to_old_entity[label] for label in new_labels]
                    if save_img:
                        result = self.grounder_model.overlay_entity_names(result, pred, entire_entities,
                                                                          custom_labels=old_labels,
                                                                          text_size=0.8,
                                                                          text_offset=-25,
                                                                          text_offset_original=-40,
                                                                          text_pixel=2)
                        imsave(result, caption, output_path, index)
                    groundings, origin_groundings = get_grounding_and_label(pred, new_labels, new_entity_to_id,
                                                                            new_to_old_entity, percent=True)
                    if output_decorator:
                        total_groundings.append(output_decorator(groundings, index, origin_groundings))
                    else:
                        total_groundings.append((groundings, index, origin_groundings))
            else:
                for pred, new_entity_to_id, new_to_old_entity, index in zip(preds, new_entity_to_ids,
                                                                            new_to_old_entities, image_ids):
                    new_labels = get_label_names(pred, self.grounder_model, entire_entities)
                    groundings, origin_groundings = get_grounding_and_label(pred, new_labels, new_entity_to_id,
                                                                            new_to_old_entity, percent=True)
                    if output_decorator:
                        total_groundings.append(output_decorator(groundings, index, origin_groundings))
                    else:
                        total_groundings.append((groundings, index, origin_groundings))
        return total_groundings
