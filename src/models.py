#!/usr/bin/env python3
"""
LSTM with POS instead of text to determine if a given text is human or LLM
generated.

Developer: Marco Cardoso
email: marcoacardosom@gmail.com
"""

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc_0 = nn.Linear(hidden_size, 16)
        self.drop_0 = nn.Dropout()
        self.act_0 = nn.ReLU()

        self.fc_1 = nn.Linear(16, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # save last hidden state

        out = self.fc_0(out)  # MLP
        out = self.drop_0(out)
        out = self.act_0(out)

        out = self.fc_1(out)  # Use the last hidden state for classification
        return out


class TextDataset(Dataset):
    def __init__(self, data, labels):
        """
        Parameters
        ----------
        data : list
            each element is a POS sequence is a numpy array of one-hot encoded
            POS Tags.
            Shape of each array: (length of sequence, size of vocabulary)
        labels : np.array
            labels for each sequence
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        _data = self.data[index]
        _labels = self.labels[index]
        return (_data, _labels)


"""
Multiple LSTMs
"""
class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return output[:, -1, :]  # Get the last output for each sequence


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc_0 = nn.Linear(input_size, hidden_size)
        self.drop_0 = nn.Dropout()
        self.act_0 = nn.ReLU()

        self.fc_1 = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        out = self.fc_0(x)
        out = self.drop_0(out)
        out = self.act_0(out)

        out = self.fc_1(out)

        return out


class TextSequenceModel(nn.Module):
    def __init__(self, lstm_input_sizes, lstm_hidden_size, lstm_num_layers, mlp_hidden_size, num_classes):
        super(TextSequenceModel, self).__init__()

        # Create a list of LSTM encoders for each input size
        self.lstm_encoders = nn.ModuleList([
            LSTMEncoder(input_size, lstm_hidden_size, lstm_num_layers)
            for input_size in lstm_input_sizes])

        # Calculate the total input size for the MLP classifier based on the number of LSTMs used
        total_input_size = lstm_hidden_size * len(self.lstm_encoders)

        # Shared MLP classifier
        self.mlp_classifier = MLPClassifier(total_input_size, mlp_hidden_size, num_classes)

    def forward(self, inputs):
        print('here')
        lstm_outputs = []

        # Pass each input through its corresponding LSTM encoder
        for i, input_data in enumerate(inputs):
            lstm_output = self.lstm_encoders[i](input_data)
            lstm_outputs.append(lstm_output)

        # Concatenate the outputs of all LSTMs
        combined_output = torch.cat(lstm_outputs, dim=1)

        # Pass the combined output through the shared MLP classifier
        output = self.mlp_classifier(combined_output)

        return output


class MultipleTextDataset(Dataset):
    def __init__(self, data, labels):
        """
        Parameters
        ----------
        data : list
            each element is a POS sequence is a numpy array of one-hot encoded
            POS Tags.
            Shape of each array: (length of sequence, size of vocabulary)
        labels : np.array
            labels for each sequence
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        retrieved_data = []

        # iterate over all representations, add to list corresponding indeces
        for _data in self.data:
            retrieved_data.append(_data[index])

        _labels = self.labels[index]
        return (tuple(retrieved_data), _labels)


"""
Helper functions
"""
def custom_collate(batch):
    # Sort batch by sequence length (descending order) for padding efficiency
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    data, labels = zip(*batch)

    # Find the length of the longest sequence in the batch
    max_length = max([x.shape[0] for x in data])

    # Initialize the padded_data array with zeros
    padded_data = np.zeros((len(data), max_length, data[0].shape[1]),
                           dtype=np.float32)

    # Pad sequences and store them in the padded_data array
    for i, seq in enumerate(data):
        padded_data[i, :seq.shape[0], :] = seq

    # Convert padded_data and labels to PyTorch tensors
    data = torch.tensor(padded_data)
    labels = torch.tensor(labels)

    return data, labels


def multiple_collate(batch):
    batch.sort(key=lambda x: x[0][0].shape[0], reverse=True)
    # split representations
    representations, labels = zip(*batch)

    # Process the representations to handle variable sequence lengths
    # Pad each representation to its maximum sequence length
    padded_representations = []
    for representation_set in representations:
        # Find the length of the longest sequence in the batch
        max_length = max([x.shape[0] for x in representation_set])

        # Initialize the padded_data array with zeros
        padded_data = np.zeros((len(representation_set), max_length,
                                representation_set[0].shape[1]), dtype=np.float32)

        # Pad sequences and store them in the padded_data array
        for i, seq in enumerate(representation_set):
            padded_data[i, :seq.shape[0], :] = seq

            # Convert padded_data and labels to PyTorch tensors
            data = torch.tensor(padded_data)
            labels = torch.tensor(labels)


        padded_data = torch.tensor(padded_data)
        padded_representations.append(padded_data)

    labels = torch.tensor(labels)

    return (tuple(padded_representations), labels)


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, device):

    for epoch in range(num_epochs):
        total_loss = 0.0

        model.train()
        for data, labels in train_loader:
            optimizer.zero_grad()

            data = data.to(device=device)
            labels = labels.to(device=device)

            # Forward pass
            outputs = model(data)
            if outputs.size()[0] != 1:
                outputs = torch.squeeze(outputs)
            else:
                outputs = outputs.view(-1)

            labels = labels.double()

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        val_loss = 0.0
        val_acc = 0.0
        amount_val = 0.0
        all_outs = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(device=device)
                labels = labels.to(device=device)

                # Forward pass
                outputs = model(data)
                outputs = torch.squeeze(outputs)
                labels = labels.double()

                # Calculate the loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _outputs = nn.functional.sigmoid(outputs)
                mask = _outputs < 0.5
                _outputs[mask] = 0
                _outputs[~mask] = 1

                all_outs.extend(_outputs.tolist())
                all_labels.extend(labels.tolist())

                mask = _outputs == labels
                val_acc += torch.sum(mask)
                amount_val += labels.size(0)


        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(all_labels, all_outs, average='macro')
        val_acc = val_acc / amount_val

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f} || "
              + f"val_loss: {avg_val_loss:.4f}, val_acc: {val_acc:.4f}, "
              + f"val_f1: {val_f1:.4f}")

    print("Training finished.")


def one_hot_encode(data, test_data):
    """
    one-hot encode a list of data

    Parameters
    ----------
    data : A list of text sequences
    test_data : A list of text sequences
    """

    # create a list, each row's POS is an element of the list
    pos_seq = data
    pos_seq_test = test_data

    # each previous element is now a list of POS tags
    split_seq = [s.split() for s in pos_seq]
    flatten_seq = [val for sublist in split_seq for val in sublist]

    split_seq_test = [s.split() for s in pos_seq_test]
    # flatten_seq_test = [val for sublist in split_seq_test for val in sublist]

    # get an array of unique POS tags in the training data == VOCABULARY
    set_seq = np.unique(flatten_seq).reshape(-1, 1)  # unique pos tags

    # one-hot encode each of the unique POS Tags
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(set_seq)

    # Create a dictionary to map values to one-hot encoded vectors
    mapping = {val: one_hot_encoded[i] for i, val in enumerate(set_seq.flatten())}

    """
    One-hot encode each sequence

    split_seq = list; each element is a list of POS Tags (correspond to a row
    in ds)
    seq = list of POS Tags, a row of ds

    In the end, `one_hot_sequences` is a list of lists.
    - Each element of the outer list corresponds to a row of ds
    - Each of this lists is itself a list of one-hot array representations of
    the pos tags.
    """
    one_hot_sequences = [[mapping[val] for val in seq] for seq in split_seq]
    # one_hot_sequences_test = [[mapping[val] for val in seq]
    #                           for seq in split_seq_test]
    one_hot_sequences_test = []
    for seq in split_seq_test:
        temp_list = []
        for val in seq:
            _value = mapping.get(val)
            if _value is not None:
                temp_list.append(_value)
            else:
                temp_list.append(np.zeros((encoder.n_features_in_,)))
        one_hot_sequences_test.append(temp_list)

    """
    change the list of arrays to a single array.
    Each element of one_hot_list_arr is an array, each row of the array corresponds
    to the one-hot representation of a POS Tag.
    The text sequence traverses in the index=0 dimension
    """
    one_hot_list_arr = [np.array(seq) for seq in one_hot_sequences]
    one_hot_list_arr_test = [np.array(seq) for seq in one_hot_sequences_test]

    return (one_hot_list_arr, one_hot_list_arr_test)
