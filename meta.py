import urllib
from newspaper import Article
from data import *

def meta_extract(url):
    article = Article(url)
    article.download()
    article.parse()
    article.download('punkt')
    article.nlp()

    return article.authors, article.publish_date, article.top_image, article.images, article.title, article.summary

# meta_extract(url)


# In[84]:


from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer, SentimentIntensityAnalyzer
from nltk.sentiment.util import *
import nltk
# nltk.download('punkt')

def sentiment(inp_text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(inp_text)


def format_raw_text(token):
    token = token.replace(' ', 'uxd')
    clean_token = ''.join(chr for chr in token if chr.isalnum() and chr.isalpha())
    clean_token = clean_token.lower()
    return clean_token