from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer, SentimentIntensityAnalyzer
from nltk.sentiment.util import *
import nltk

def sentiment(inp_text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(inp_text)