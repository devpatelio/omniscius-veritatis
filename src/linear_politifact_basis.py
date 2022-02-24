from data import data_dict, DIRECTORY
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import nltk
import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from keras.preprocessing import sequence
import keras
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split


class PreprocessingDataset(Dataset):
    def __init__(self, file, root, x_col, y_col, meta_columns, label_idx = -1):
        self.x_col = x_col
        self.y_col = y_col
        self.data = pd.read_csv(file)
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.data = self.data.drop(meta_columns, axis=1)

        # self.data, self.base_ref = self.tokenizer(self.data, [x_col])
        self.x_data = self.data[x_col]
        # self.max_len = max([len(i) for i in self.x_data])
        self.max_len = 600

        self.x_data, self.tokenizer = self.word_vector(self.x_data)
        self.data[x_col] = [torch.FloatTensor(i) for i in self.x_data]
        self.data = self.vectorize(self.data, [y_col])
        self.df_data = self.data
        self.data = self.data.to_numpy()

        self.root = root
        self.transform = transforms.Compose([transforms.ToTensor()])

    def format_text(self, token):
        clean_token = ''.join(chr for chr in token if chr.isalnum() and chr.isalpha())
        return clean_token

    def word_vector(self, data):
        x_data = data
        x_data = list(x_data)
        maximum_length = 0
        max_idx = 0
        for idx, i in enumerate(x_data):

            if len(i) > maximum_length:
                maximum_length = len(i)
                max_idx = idx
        
        maximum_length = 600
        t = Tokenizer(num_words=maximum_length)
        t.fit_on_texts(x_data)
        sequences = t.texts_to_sequences(x_data)
        sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maximum_length)
        # print(x_data[0])
        # print(len(x_data[0]))
        # print(sequences[0])

        return sequences, t


    def vectorize(self, data_inp, columns):
        data = data_inp
        for column in columns:
            labels = list(data[column].unique())
            ref = dict(zip(data[column].unique(), [i for i in range(len(labels))]))
            # print(ref)
            for idx, val in enumerate(data[column]):
                vectorized = ref[data[column][idx]]
                data[column][idx] = torch.tensor(vectorized, dtype=float)
        return data

    def __len__ (self):
        return len(self.data)

    def __getitem__ (self, idx):
        
        self.transpose_data = self.data
        self.transpose_data = self.transpose_data.transpose()
        x_data = self.transpose_data[0]
        y_data = self.transpose_data[1]

        return x_data[idx], y_data[idx]

clean_truth_data = PreprocessingDataset(data_dict['politifact_clean_binarized'], DIRECTORY, 'statement', 'veracity', ['source', 'link'])

token_basis = clean_truth_data.tokenizer

import random
BATCH_SIZE = 64

primary_data = clean_truth_data #secondary option of truth_data

train_len = int(len(primary_data)*0.8)
test_len = len(primary_data) - train_len

train_set, test_set = torch.utils.data.random_split(primary_data, [train_len, test_len])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# print(len(train_set))
# print(len(test_set))
# print(len(primary_data))

num_feats = np.array([train_set[i][0]for i in range(len(train_set))])
num_labels = np.array([train_set[i][1]for i in range(len(train_set))])


# print(num_feats.shape)
# print(num_labels.shape)

if primary_data == clean_truth_data:
    a = iter(train_loader)
    b = np.array(next(a))
    inp_size = (b[0].shape)[1]
else:
    inp_size = str(num_feats[0].shape)[-5:-2]

print(f"Input_size: {inp_size}")

def convert_word_to_token(tokener, sentence, max_len=600):
    x_data = sentence

    sequences = tokener.texts_to_sequences(x_data)
    sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)

    return sequences

