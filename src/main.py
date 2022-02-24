import torch
import torch.nn
from torch.nn import functional as F
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
from data import clean_truth_data
import itertools
from net import FeedForward, RecurrentClassifier
from meta_article import tokenize_sequence, prediction
from data import DIRECTORY
from meta_article import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
primary_data = clean_truth_data #secondary option of truth_data

train_len = int(len(primary_data)*0.8)
test_len = len(primary_data) - train_len

train_set, test_set = torch.utils.data.random_split(primary_data, [train_len, test_len])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

num_feats = np.array([train_set[i][0]for i in range(len(train_set))])
num_labels = np.array([train_set[i][1]for i in range(len(train_set))])

a = iter(train_loader)
b = next(a)
b = np.asarray(b)
inp_size = (b[0].shape)[1]


ab = list(itertools.chain(*[i[0] for i in clean_truth_data]))
ab = set([int(i) for i in ab])
emb_dim = len(ab)

max_len = len(train_set[1][0])
ref_check = 1

def train(net, train_loader, LR, DECAY, EPOCHS):
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=DECAY)
    loss_func = torch.nn.BCEWithLogitsLoss()

    epochs = EPOCHS
    losses = []

    for step in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inp, labels = data
            if net == recurrent:
                inp, labels = inp.long().to(device), labels.float().to(device)
                optimizer.zero_grad()
                outputs = net(inp)
                cost = loss_func(torch.squeeze(outputs), torch.squeeze(labels))
            elif net == feedforward:
                inp, labels = inp.float().to(device), labels.float().to(device)
                optimizer.zero_grad()
                outputs = net(inp)
                cost = loss_func(torch.squeeze(outputs), labels)
            cost.backward()
            optimizer.step()

            running_loss += cost.item()
        print(f'Epoch: {step}   Training Loss: {running_loss/len(train_loader)}')
    print('Training Complete')  

    return losses

def eval(net, test_loader):
    total = 0
    acc = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=DECAY)

    for i, data in enumerate(test_loader):
        inp, labels = data
        optimizer.zero_grad()
        output = net(inp.float())
        output = output.detach().numpy()
        output = list(output)
        output = [list(i).index(max(i)) for i in output]
        
        for idx, item in enumerate(torch.tensor(output)):
            total += 1
            if item == labels[idx]:
                acc += 1
    print(f'{acc/total*100}%')

feedforward = FeedForward(ref_check, inp_size).to(device)
recurrent = RecurrentClassifier(emb_dim, inp_size, 50, ref_check, 2, dropout=0.2).to(device)

def model_load(net, PATH, name, export=True):
    if export:
        torch.save(net.state_dict(), PATH+name+'.pth')
        return PATH+name+'.pth'
    else:
        net.torch.load_state_dict(torch.load(PATH + name + '.pth'))
        return net
    
PATH = '/Users/devpatelio/Downloads/Coding/Global_Politics_EA/'
# train(feedforward, train_loader, 1e-3, 5e-3, 200)
model_load(feedforward, str(PATH+'model_parameters/'), 'linear_politifact')

# train(recurrent, train_loader, 1e-3, 5e-3, 200)
model_load(recurrent, str(PATH+'model_parameters/'), 'lstm_politifact')

token_basis = clean_truth_data.token



# inp = tokenize_sequence(summ, token_basis)
# inp = inp[None, :]
# print(inp.shape)

