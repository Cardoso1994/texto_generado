#!/usr/bin/env python3
"""
LSTM with POS instead of text to determine if a given text is human or LLM
generated.

Developer: Marco Cardoso
email: marcoacardosom@gmail.com
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, \
    SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence

from models import TextDataset, LSTMClassifier, train_model, custom_collate, \
    one_hot_encode


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
POS = 'xpos'
LR = 5e-4
# LR = 2e-4
BATCH_SIZE = 64


ds = pd.read_csv("../text/train_c_subtask1.csv")
ds_test = pd.read_csv("../text/test_preprocess_labels.csv")

# create a list, each row's POS is an element of the list
pos_seq = ds[POS].tolist()
pos_seq_test = ds_test[POS].tolist()

pos_data, pos_test_data = one_hot_encode(pos_seq, pos_seq_test)

labels = ds['num_label'].to_numpy()
labels_test = ds_test['num_label'].to_numpy()

size_of_ds = labels.shape[0]
idx = np.arange(size_of_ds, dtype='int')

val_split = int(0.1 * size_of_ds)
# test_split = int(0.1 * size_of_ds)
val_idx = idx[:val_split]
train_idx = idx[val_split:]

dataset = TextDataset(pos_data, labels)
dataset_test = TextDataset(pos_test_data, labels_test)


# Create random samplers for training and validation
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                          collate_fn=custom_collate, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                          collate_fn=custom_collate, sampler=train_sampler)

test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE,
                         collate_fn=custom_collate)

input_size = pos_data[0].shape[1]
# hidden_size = 64
hidden_size = 32
num_layers = 1
output_size = 1
num_epochs = 80

# Create the model and set up loss function and optimizer
model = LSTMClassifier(input_size, hidden_size, num_layers,
                       output_size).to(device=DEVICE)
criterion = nn.BCEWithLogitsLoss().to(device=DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs,
            DEVICE)

# Test the model
test_acc = 0.0
ammount_test = 0.0
all_outs = []
all_labels = []
model.eval()
with torch.no_grad():
    for data, labels in test_loader:
        data = data.to(device=DEVICE)
        labels = labels.to(device=DEVICE)

        # Forward pass
        outputs = model(data)
        outputs = torch.squeeze(outputs)
        labels = labels.double()

        _outputs = nn.functional.sigmoid(outputs)
        mask = _outputs < 0.5
        _outputs[mask] = 0
        _outputs[~mask] = 1

        all_outs.extend(_outputs.tolist())
        all_labels.extend(labels.tolist())

        mask = _outputs == labels
        test_acc += torch.sum(mask)
        ammount_test += labels.size(0)

test_f1 = f1_score(all_labels, all_outs, average='macro')
test_acc = test_acc / ammount_test

print(f"test_acc: {test_acc:.4f}, " + f"val_f1: {test_f1:.4f}")

print("Testing finished.")
