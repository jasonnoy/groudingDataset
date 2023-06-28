import webdataset as wds
url = "/gpfs/gpfs1/zphz/img_datasets/laion115m/part-00001/100045.tar"
dataset = wds.WebDataset(url)

for sample in dataset:
    break
print(type(sample))
for k,v in sample.items():
    if k == "image":
        print(v.nbytes)
    print(k, v)