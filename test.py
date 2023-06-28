import webdataset as wds
url = "/gpfs/gpfs1/zphz/img_datasets/laion115m/part-00001/100045.tar"
dataset = wds.WebDataset(url).shuffle(1000).decode("rgb").to_tuple("png", "json")

for image, json in dataset:
    break
print(json)