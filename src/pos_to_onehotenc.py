#!/usr/bin/env python3
"""
Receives a corpus with POS Taggings and encodes them via one hot
"""

import pickle

import numpy as np
import pandas as pd

POS = 'xpos'  # xpos or upos

ds = pd.read_csv("../text/train_c.csv")
text = ds[POS].to_list()

pos_tags = set()

for seq in text:
    seq = seq.split()
    pos_tags.update(seq)

print(len(pos_tags))

pos_to_index = {pos: index for index, pos in enumerate(pos_tags)}

# Open a file and use dump()
with open(f'./{POS}_onehotenc.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(pos_to_index, file)
