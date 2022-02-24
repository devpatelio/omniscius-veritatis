import torch
import torch.nn
from torch.nn import functional as f
import pandas as pd
import json
import numpy as np
import os
from keras.preprocessing import sequence
import keras
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
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

DIRECTORY = 'data'
paths = []

for root, dirs, files in os.walk(DIRECTORY):
    for name in files:
        paths.append(os.path.join(root, name))

names = [i.split('/')[-1] for i in paths][1:]
data_dict = dict(zip([i[:-4] for i in names], paths[1:]))


class PreprocessingDataset(Dataset):
    def __init__(self, file, root, x_col, y_col, meta_columns, label_idx = -1):
        self.x_col = x_col
        self.y_col = y_col
        self.data = pd.read_csv(file)
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.data = self.data.drop(meta_columns, axis=1)

        # self.data, self.base_ref = self.tokenizer(self.data, [x_col])
        self.x_data = self.data[x_col]
        self.max_len = max([len(i) for i in self.x_data])
        self.max_len = 600

        self.x_data, self.token = self.word_vector(self.x_data)
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
        t = Tokenizer(num_words=600, filters='\n.,:!"#$()&@%^()-_`~[];.,{|}')
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