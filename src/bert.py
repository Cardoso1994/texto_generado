#!/usr/bin/env python3

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler, \
    SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW


torch.cuda.empty_cache()

""" GLOBAL VARIABLES """
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42
LR = 1.5e-6
BATCH_SIZE = 32
# MAX_LEN = 512
MAX_LEN = 256
WEIGHT_DECAY = 0.01


torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# read training data
ds = pd.read_csv("../text/train_c.csv")
print(ds.columns)
# text = ds['text'].to_list()
text = ds['upos'].to_list()
text = [s.lower() for s in text]

# read test data
# test_ds = pd.read_csv("../text/test.tsv", sep='\t')
# # test_text = test_ds['text'].to_list()
# test_text = test_ds['upos'].to_list()
# test_text = [s.lower() for s in test_text]
# test_ids = test_ds['id'].to_list()

"""
classes
generated: 1; human: 0
"""
labels = ds['label']
mask_gen = labels == 'generated'
mask_hum = labels == 'human'
labels[mask_gen] = 1
labels[mask_hum] = 0
labels = labels.astype('int')
labels = torch.tensor(labels.to_list())

""" tokenizer - preprocess """
# training tokens
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer.batch_encode_plus(
    text,
    add_special_tokens=True,
    max_length=MAX_LEN,  # best so far: 128
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_token_type_ids=True,
    return_tensors='pt')

# extract the input IDs, attention masks and token type IDs
input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']
token_type_ids = tokens['token_type_ids']

# testing tokens
# test_tokens = tokenizer.batch_encode_plus(
#     test_text,
#     add_special_tokens=True,
#     max_length=MAX_LEN,  # best so far: 128
#     padding='max_length',
#     truncation=True,
#     return_attention_mask=True,
#     return_token_type_ids=True,
#     return_tensors='pt')

# extract the input IDs, attention masks and token type IDs
# test_input_ids = tokens['input_ids']
# test_attention_mask = tokens['attention_mask']
# test_token_type_ids = tokens['token_type_ids']

# model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased').to(DEVICE)

# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# training
# create train-val-test splits from training dataset
num_samples = len(text)
idx = np.arange(num_samples)
split_val = int(num_samples * 0.2)
split_test = int(num_samples * 0.1)

ds_train_idx = idx[split_val:]
ds_val_idx = idx[split_test:split_val]
ds_test_idx = idx[:split_test]

# idx of test dataset
# test_ds_num_samples = len(test_text)
# test_ds_idx = np.arange(test_ds_num_samples)

train_sampler = SubsetRandomSampler(ds_train_idx)
val_sampler = SubsetRandomSampler(ds_val_idx)
test_sampler = SubsetRandomSampler(ds_test_idx)
# test_ds_sampler = SequentialSampler(test_ds_idx)

dataset = TensorDataset(input_ids, attention_mask, labels)
# test_dataset = TensorDataset(test_input_ids, test_attention_mask)

train_dl = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_dl = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
test_dl = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler)
# test_ds_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE,
#                         sampler=test_ds_sampler)

# training
num_epochs = 20
no_improvement = 0

best_loss = float('inf')
print("Start of training")
for epoch in range(num_epochs):
    model.train()

    train_loss = 0.
    y_pred_tr = []
    y_true_tr = []
    for batch in train_dl:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1)
        y_pred = torch.argmax(probs, dim=1).tolist()
        y_true = labels.cpu().tolist()
        y_pred_tr += y_pred
        y_true_tr += y_true
        __loss = outputs.loss
        __loss.backward()
        optimizer.step()

        train_loss += __loss

    val_loss = 0.
    y_pred_val = []
    y_true_val = []
    model.eval()
    with torch.no_grad():
        for batch in val_dl:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask,
                            labels=labels)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            y_pred = torch.argmax(probs, dim=1).tolist()
            y_true = labels.cpu().tolist()
            y_pred_val += y_pred
            y_true_val += y_true

            __loss = outputs.loss
            val_loss += __loss

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                    "./best_model.pt")
            no_improvement = 0

        if no_improvement == 5:
            break
        no_improvement += 1

    f1_tr = f1_score(y_true_tr, y_pred_tr, average='macro')
    f1_val = f1_score(y_true_val, y_pred_val, average='macro')
    ys = {'f1_tr': f1_tr, 'f1_val': f1_val}

    with open(f'ys_len{MAX_LEN}.pkl', 'wb') as f:
        pickle.dump(ys, f)

    if epoch % 2 == 0:
        print(f"Epoch: {epoch}, train_loss: {train_loss}, "
                +f"f1_tr: {f1_tr:.2f}, "
                + f"val_loss: {val_loss}, f1_val: {f1_val}")

checkpoint = torch.load('./best_model.pt')
print(f"best model found at epoch: {checkpoint['epoch']}")
model.load_state_dict(checkpoint['model_state_dict'])

# testing
torch.cuda.empty_cache()
model.eval()
y_true_test = []
y_pred_test = []
y_pred_test_ds = []
with torch.no_grad():
    # testing on test partition from training dataset
    for batch in test_dl:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        y_true = labels.cpu().tolist()
        y_pred = torch.argmax(probs, dim=1).tolist()
        y_pred_test += y_pred
        y_true_test += y_true
    f1_test = f1_score(y_true_test, y_pred_test, average='macro')

    # assigning labels to competition's test set
    # y_pred_test_ds = []
    # for batch in test_ds_dl:
    #     input_ids, attention_mask = batch
    #     input_ids = input_ids.to(DEVICE)
    #     attention_mask = attention_mask.to(DEVICE)

    #     outputs = model(input_ids, attention_mask=attention_mask)
    #     probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    #     y_pred = torch.argmax(probs, dim=1).tolist()
    #     y_pred_test_ds += y_pred

print()
print(f"Max Length: {MAX_LEN}, Test:  f1_score = {f1_test}")
# test_labels = ['human' if elem == 0 else 'generated'
#                 for elem in y_pred_test_ds]

# test_df = {'id': test_ids, 'label': test_labels}
# test_df = pd.DataFrame(test_df)
# test_df.to_csv(f'./data_frame_{MAX_LEN}.tsv', sep='\t', index=False,
#                index_label=False)

# Open the file in binary mode and read the list from it
# with open(f'test_labels_len{MAX_LEN}.pkl', 'wb') as f:
#     pickle.dump(test_labels, f)
