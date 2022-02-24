
import urllib
from newspaper import Article
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer, SentimentIntensityAnalyzer
from nltk.sentiment.util import *
import nltk
from data import *
import torch

def meta_extract(url):
    article = Article(url)
    article.download()
    article.parse()
    article.download('punkt')
    article.nlp()
    return article.authors, article.publish_date, article.top_image, article.images, article.title, article.summary

def sentiment(inp_text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(inp_text)


def tokenize_sequence(text_inp, tokenizer):
    text_inp = text_inp.lower().split('\n')
    tokenizer.fit_on_texts(text_inp)
    sequences = tokenizer.texts_to_sequences(text_inp)
    sequences = [i if i!=[] else [0] for i in tokenizer.texts_to_sequences(text_inp)]
    sequences = [i[0] for i in sequences]
    pad_len =  [0]*int(inp_size - len(sequences))
    sequences += pad_len
    return torch.FloatTensor(sequences)[:600]


def prediction(inp, model):
    output = model(inp)
    return output