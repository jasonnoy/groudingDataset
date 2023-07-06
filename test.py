import json
import os
import webdataset as wds
from PIL import Image
import io


def read_tar(tar_path):
    return wds.WebDataset(tar_path)


if __name__ == "__main__":
    # nlp = spacy.load("en_core_web_trf")
    input_path = "/gpfs/gpfs1/zphz/img_datasets/laion115m/part-00032"
    output_path = "/gpfs/gpfs1/zphz/jjh/test_dataset/part-00032"
    ids = [0, 1, 2, 3, 4, 5]
    for idx in ids:
        res = {}
        tar_filename = "{}.tar".format(3200000+idx)
        tar_dataset = read_tar(os.path.join(input_path, tar_filename))
        meta_filename = "{}.meta.jsonl".format(3200000+idx)
        with open(os.path.join(input_path, meta_filename), 'r', encoding='utf-8') as f1, open(os.path.join(output_path, meta_filename), 'a', encoding='utf-8') as f2:
            for data, line in zip(tar_dataset, f1):
                meta_data = json.loads(line)
                if meta_data['status'] == "success":
                    size = (int(meta_data['width']), int(meta_data['height']))
                    index = data['id'].decode()
                    sample_id = meta_data['SAMPLE_ID']
                    assert (str(index) == str(sample_id))
                    image_b = data['jpg']
                    image = Image.open(io.BytesIO(image_b)).convert('RGB')
                    image.save("test.png", format="png")
                    caption = data['txt']

                    ret = parse_and_grounding_multi_class(image, caption, str(idx), nlp, output_path, True)
                    meta_data.update(ret)
                f2.write(json.dumps(meta_data, ensure_ascii=False) + '\n')
                break
        f1.close()
        f2.close()
        break
    print("done")
