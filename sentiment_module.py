import os
import nltk
import random
import string
from nltk.tokenize import word_tokenize
from sklearn.externals import joblib

with open(os.getcwd() + "/data/documents.pickle","rb") as f:
    documents = joblib.load(f)

def find_features(doc):
    words = word_tokenize(doc)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

with open(os.getcwd() + "/data/word_features.pickle","rb") as f:
    word_features = joblib.load(f)


with open(os.getcwd() + "/data/original_classifier.pickle","rb") as f:
    classifier = joblib.load(f)

def sentiment(text):
    feats  = find_features(text)
    return classifier.classify(feats)
