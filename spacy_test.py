import argparse
import torch
import json
import os
import sys
import spacy
from tqdm import tqdm
import webdataset
import warnings
import math
import spacy

nlp = spacy.load("en_core_web_trf")

text = "Apple is looking at buying U.K. startup for $1 billion"
doc = nlp(text)
for t in doc.noun_chunks:
    print(t.text, len(t.text))
    print(t[0].text)
    print(t[0].idx)