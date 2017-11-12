import os
import nltk
import random
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import movie_reviews, stopwords

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from nltk.classify import ClassifierI
from statistics import mode
from sklearn.externals import joblib

stop = stopwords.words('english') + list(string.punctuation)

short_pos = open(os.getcwd() + '/data/positive.txt', 'r').read()
short_neg = open(os.getcwd() + '/data/negative.txt', 'r').read()

short_pos = unicode(short_pos, errors='ignore')
short_neg = unicode(short_neg, errors='ignore')

documents = []

for p in short_pos.split('\n'):
    documents.append((p, 'pos'))

for p in short_neg.split('\n'):
    documents.append((p, 'neg'))

with open(os.getcwd() + "/data/documents.pickle","wb") as f:
    joblib.dump(documents, f)

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

all_words = []
for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

with open(os.getcwd() + "/data/word_features.pickle","wb") as f:
    joblib.dump(word_features, f)

def find_features(doc):
    words = word_tokenize(doc)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featureset = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featureset)

# training and testing sample
training_set = featureset[:10000]
testing_set = featureset[10000:]

print('classification in process for original classifier')
classifier = nltk.NaiveBayesClassifier.train(training_set)
print('original accuracy : {}'.format(nltk.classify.accuracy(classifier ,testing_set)*100))
classifier.show_most_informative_features(15)
with open(os.getcwd() + "/data/original_classifier.pickle","wb") as f:
    joblib.dump(classifier, f)
