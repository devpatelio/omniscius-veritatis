#!/usr/bin/env python
# coding: utf-8

# In[82]:


# import torch
# import torch.nn
# from torch.nn import functional as f
# import pandas as pd
# import json
# import numpy as np
# import os

# DIRECTORY = 'data'
# paths = []

# for root, dirs, files in os.walk(DIRECTORY):
#     for name in files:
#         paths.append(os.path.join(root, name))

# names = [i.split('/')[-1] for i in paths][1:]
# data_dict = dict(zip([i[:-4] for i in names], paths[1:]))


# In[83]:


# import urllib
# from newspaper import Article
# from data import *

# def meta_extract(url):
#     article = Article(url)
#     article.download()
#     article.parse()
#     article.download('punkt')
#     article.nlp()

#     return article.authors, article.publish_date, article.top_image, article.images, article.title, article.summary

# # meta_extract(url)


# # In[84]:


# from nltk.classify import NaiveBayesClassifier
# from nltk.corpus import subjectivity
# from nltk.sentiment import SentimentAnalyzer, SentimentIntensityAnalyzer
# from nltk.sentiment.util import *
# import nltk

# def sentiment(inp_text):
#     sia = SentimentIntensityAnalyzer()
#     return sia.polarity_scores(inp_text)


# In[104]:


from data import *
from meta import *
from net import *


# class PreprocessingDataset(Dataset):
#     def __init__(self, file, root, x_col, y_col, meta_columns, label_idx = -1):
#         self.x_col = x_col
#         self.y_col = y_col
#         self.data = pd.read_csv(file)
#         self.data = self.data.sample(frac=1).reset_index(drop=True)
#         self.data = self.data.drop(meta_columns, axis=1)

#         # self.data, self.base_ref = self.tokenizer(self.data, [x_col])
#         self.x_data = self.data[x_col]
#         self.max_len = max([len(i) for i in self.x_data])
#         self.max_len = 600

#         self.x_data, self.token = self.word_vector(self.x_data)
#         self.data[x_col] = [torch.FloatTensor(i) for i in self.x_data]
#         self.data = self.vectorize(self.data, [y_col])
#         self.df_data = self.data
#         self.data = self.data.to_numpy()

#         self.root = root
#         self.transform = transforms.Compose([transforms.ToTensor()])

#     def format_text(self, token):
#         clean_token = ''.join(chr for chr in token if chr.isalnum() and chr.isalpha())
#         return clean_token

#     def word_vector(self, data):
#         x_data = data
#         x_data = list(x_data)
#         maximum_length = 0
#         max_idx = 0
#         for idx, i in enumerate(x_data):

#             if len(i) > maximum_length:
#                 maximum_length = len(i)
#                 max_idx = idx
#         maximum_length = 600
#         t = Tokenizer(num_words=600, filters='\n.,:!"#$()&@%^()-_`~[];.,{|}')
#         t.fit_on_texts(x_data)
#         sequences = t.texts_to_sequences(x_data)
#         sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maximum_length)
#         print(x_data[0])
#         print(len(x_data[0]))
#         print(sequences[0])

#         return sequences, t


#     def vectorize(self, data_inp, columns):
#         data = data_inp
#         for column in columns:
#             labels = list(data[column].unique())
#             ref = dict(zip(data[column].unique(), [i for i in range(len(labels))]))
#             print(ref)
#             for idx, val in enumerate(data[column]):
#                 vectorized = ref[data[column][idx]]
#                 data[column][idx] = torch.tensor(vectorized, dtype=float)
#         return data

#     def __len__ (self):
#         return len(self.data)

#     def __getitem__ (self, idx):
        
#         self.transpose_data = self.data
#         self.transpose_data = self.transpose_data.transpose()
#         x_data = self.transpose_data[0]
#         y_data = self.transpose_data[1]

#         return x_data[idx], y_data[idx]

DIRECTORY = 'data'
data_dict = {'politifact': 'data/truth-detectiondeception-detectionlie-detection/politifact.csv', 'politifact_clean': 'data/truth-detectiondeception-detectionlie-detection/politifact_clean.csv', 'politifact_clean_binarized': 'data/truth-detectiondeception-detectionlie-detection/politifact_clean_binarized.csv'}
clean_truth_data = PreprocessingDataset(data_dict['politifact_clean_binarized'], DIRECTORY, 'statement', 'veracity', ['source', 'link'])

BATCH_SIZE = 64

primary_data = clean_truth_data #secondary option of truth_data

train_len = int(len(primary_data)*0.8)
test_len = len(primary_data) - train_len

train_set, test_set = torch.utils.data.random_split(primary_data, [train_len, test_len])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# print(len(train_set))
# print(len(test_set))

num_feats = np.array([train_set[i][0].numpy() for i in range(len(train_set))])
num_labels = np.array([train_set[i][1] for i in range(len(train_set))])

a = iter(train_loader)
b = next(a)
b = np.asarray(b[0])
# print(b.shape)
inp_size = (b.shape)[1]
# print(inp_size)
# print(inp_size)


# In[87]:


import itertools
ab = list(itertools.chain(*[i[0] for i in clean_truth_data]))
# print(len(ab))
ab = set([int(i) for i in ab])
emb_dim = len(ab)



# In[105]:


# import torch.nn as nn
# import torch.nn.functional as F

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class FeedForward(nn.Module):
#     def __init__(self, num_classes, input_size, kernel_size=4):
#         super(FeedForward, self).__init__()
#         self.fc1 = nn.Linear(input_size, 200)
#         self.fc3 = nn.Linear(200, 100)        
#         self.fc4 = nn.Linear(100, 100)
#         self.fc5 = nn.Linear(100, 50)
#         self.fc6 = nn.Linear(50, 20)
#         self.fc7 = nn.Linear(20, 1)
#         self.dropout = nn.Dropout(0.3)

#     def forward(self, x):
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.dropout(F.relu(self.fc3(x)))
#         x = self.dropout(F.relu(self.fc4(x)))
#         x = self.dropout(F.relu(self.fc5(x)))
#         x = self.dropout(F.relu(self.fc6(x)))
#         x = self.dropout(F.relu(self.fc7(x)))

        
#         return x


# class RecurrentClassifier(nn.Module):
#     def __init__(self, embedding_dim, input_size, hidden_size, output_size, num_layers, dropout=0.3):
#         super(RecurrentClassifier, self).__init__()

#         self.embedding = nn.Embedding(embedding_dim, input_size)
#         self.rnn = nn.LSTM(input_size, 
#                             hidden_size,
#                             num_layers,
#                             batch_first = True,
#                             dropout=dropout)
#         self.fc1 = nn.Linear(hidden_size*2, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.embedding(x)
#         x, (hidden, cell) = self.rnn(x)
#         print(hidden.shape)
#         hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1, :, :]), dim=1))
#         x = self.fc1(hidden)
#         x = self.dropout(self.fc2(x))

#         return x



max_len = len(train_set[1][0])
ref_check = 1

feedforward = FeedForward(ref_check, inp_size).to(device)
recurrent = RecurrentClassifier(emb_dim, inp_size, 50, ref_check, 2, dropout=0.2).to(device)


# print(feedforward, recurrent)


# In[114]:



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


# In[116]:


def model_load(net, PATH, name, export=True):
    if export:
        torch.save(net.state_dict(), PATH+name+'.pth')
        return PATH+name+'.pth'
    else:
        net.torch.load_state_dict(torch.load(PATH + name + '.pth'))
        return net


# train(feedforward, train_loader, 1e-3, 5e-3, 200)
model_load(feedforward, 'model_parameters/', 'linear_politifact')

# train(recurrent, train_loader, 1e-3, 5e-3, 200)
model_load(recurrent, 'model_parameters/', 'lstm_politifact')


# In[138]:



# url = 'https://www.historyofvaccines.org/content/blog/defense-of-common-antivaxxer'

# _, _, _, _, _, summ = meta_extract(url)

token_basis = clean_truth_data.token


def tokenize_sequence(text_inp, tokenizer):
    text_inp = text_inp.lower().split('\n')
    tokenizer.fit_on_texts(text_inp)
    sequences = tokenizer.texts_to_sequences(text_inp)
    sequences = [i if i!=[] else [0] for i in tokenizer.texts_to_sequences(text_inp)]
    sequences = [i[0] for i in sequences]
    pad_len =  [0]*int(inp_size - len(sequences))
    sequences += pad_len
    return torch.FloatTensor(sequences)[:600]

# inp = tokenize_sequence(summ, token_basis)
# inp = inp[None, :]
# print(inp.shape)

def prediction(inp, model):
    output = model(inp)
    return output

# a = prediction(inp, feedforward)


# In[143]:



# feedforward_template = FeedForward(ref_check, inp_size).to(device)
# recurrent_template = RecurrentClassifier(emb_dim, inp_size, 50, ref_check, 2, dropout=0.2).to(device)    
# model_load(feedforward_template, '/Users/devpatelio/Downloads/Coding/Global_Politics_EA/model_parameters/', 'linear_politifact')
# model_load(recurrent_template, '/Users/devpatelio/Downloads/Coding/Global_Politics_EA/model_parameters/', 'lstm_politifact')

# # feedforward_template.eval()
# # recurrent_template.eval()

# output_linear = '0'
# output_lstm = '1' #check for error without passing error

# output_linear = prediction(inp, feedforward_template)
# output_lstm = prediction(inp.long(), recurrent_template)

# if output_linear == 0:
#     output_linear = f"No Bias: Prediction = {output_linear}"
# elif output_linear == 1:
#     output_linear = f"Bias: Prediction = {output_linear}"


# if output_lstm == 0:
#     output_lstm = f"Limited Veracity: Prediction = {output_lstm}"
# elif output_lstm == 1:
#     output_lstm = f"Expressive Veracity: Prediction = {output_lstm}"


# In[179]:



from django.shortcuts import render
# from scrape import meta_extract
import flask
from flask import Flask, request, render_template, redirect, url_for
# from sentiment import sentiment
import numpy as np
import pandas as pd
import json
import random
import os
import html
import torch
# from linear_politifact_basis import token_basis, convert_word_to_token
from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, redirect, url_for, request
app = Flask(__name__)
# run_with_ngrok(app)   #starts ngrok when the app is run

from flask import Flask, jsonify, request
from string import punctuation
from collections import Counter
import random


def model_load(net, PATH, name, export=True):
    if export:
        torch.save(net.state_dict(), PATH+name+'.pth')
        return PATH+name+'.pth'
    else:
        net.torch.load_state_dict(torch.load(PATH + name + '.pth'))
        net.eval()

@app.route('/')
def hello():
    return render_template("home.html")

@app.route('/link', methods=["POST", "GET"])
def link():
    if request.method == "POST":
        link_inp = request.form['linker']
        # print(type(link_inp))
    
        link_inp = link_inp.replace('.com', 'comkey')
        link_inp = link_inp.replace('https://', 'https')
        link_inp = link_inp.replace('www.', 'www')
        link_inp = link_inp.replace('/', 'slash')
        # print(link_inp)
        main = link_inp

        return redirect(url_for("preview_linker", linkage=main, tag='link_url'))
    else:
        return render_template("link.html")


@app.route('/text', methods=["POST", "GET"])
def pure_text():
    if request.method == "POST":
        inp_raw = request.form['raw_text']
        inp_raw = format_raw_text(inp_raw)
        return redirect(url_for('preview_linker', linkage=inp_raw, tag='pure_text'))
    else:
        return render_template("pure_text.html")


@app.route(f"/output/<tag>/<linkage>")
def preview_linker(linkage, tag):
    preview = linkage
    if tag == 'link_url':
        preview = preview.replace('https', 'https://')
        preview = preview.replace('www', 'www.')
        preview = preview.replace('slash', '/')
        preview = preview.replace('comkey', '.com')
        authart, publ, timg, allimg, tit, summ = meta_extract(preview)

    elif tag == 'pure_text':
        preview = preview.replace('uxd', ' ')     
        summ = preview  
        empty_msg = 'None'
        authart = empty_msg
        publ = empty_msg
        timg = empty_msg
        allimg = empty_msg
        tit = empty_msg

    sent = sentiment(summ)

    inp = tokenize_sequence(summ, token_basis)
    inp = inp[:600]
    inp = inp[None, :]
    # print(inp.shape)

    feedforward_template = FeedForward(ref_check, inp_size).to(device)
    recurrent_template = RecurrentClassifier(emb_dim, inp_size, 50, ref_check, 2, dropout=0.2).to(device)    
    model_load(feedforward_template, 'model_parameters/', 'linear_politifact')
    model_load(recurrent_template, 'model_parameters/', 'lstm_politifact')

    # feedforward_template.eval()
    # recurrent_template.eval()

    output_linear = '0 ERROR'
    output_lstm = '1 ERROR' #check for error without passing error

    output_linear = F.sigmoid(prediction(inp, feedforward_template)).round()
    output_lstm = F.sigmoid(prediction(inp.long(), recurrent_template))

    all_types = list(pd.read_csv(data_dict['politifact_clean'])['veracity'].unique())

    if output_linear == 0:
        output_linear = f"Little Bias: Prediction = {output_linear}"
    elif output_linear == 1:
        output_linear = f"Substantial Bias: Prediction = {output_linear}"

    statement_type = ''
    if output_lstm <= 0.25:
        statement_type = 'True'
    elif 0.25 < output_lstm <= 0.5:
        statement_type = 'Mostly True'
    elif 0.5 < output_lstm <= 0.75:
        statement_type = 'Mostly False'
    elif 0.75 < output_lstm <= 1:
        statement_type = 'False'
    elif output_lstm > 1:
        statement_type = 'Pants on Fire!'

    output_lstm = f"Veracity -> {statement_type}: {output_lstm}"

    # if output_lstm == 0:
    #     output_lstm = f"Limited Veracity: Prediction = {output_lstm}"
    # elif output_lstm == 1:
    #     output_lstm = f"Expressive Veracity: Prediction = {output_lstm}"

    

    

    return render_template("preview.html", preview_link=preview,
                                            author_article=authart, 
                                            published_article=publ,
                                            top_image = timg,
                                            all_image = allimg,
                                            title_article=tit,
                                            summary_article=summ,
                                            sentiment=sent,
                                            bias_point=output_linear,
                                            skew_point=output_lstm)

