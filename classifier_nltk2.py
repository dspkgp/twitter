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


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


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


# print('classification in process for MNB_classifier')
# MNB_classifier = SklearnClassifier(MultinomialNB())
# MNB_classifier.train(training_set)
# print('MNB_classifier accuracy : {}'.format(nltk.classify.accuracy(MNB_classifier ,testing_set)*100))
# with open("MNB_classifier.pickle","wb") as f:
#     joblib.dump(MNB_classifier, f)

# print('classification in process for LogisticRegression_classifier')
# LogisticRegression = SklearnClassifier(LogisticRegression())
# LogisticRegression.train(training_set)
# print('LogisticRegression accuracy : {}'.format(nltk.classify.accuracy(LogisticRegression ,testing_set)*100))
# with open("LogisticRegression.pickle","wb") as f:
#     joblib.dump(LogisticRegression, f)

# print('classification in process for SGDclassifier')
# SGDClassifier = SklearnClassifier(SGDClassifier())
# SGDClassifier.train(training_set)
# print('SGDClassifier accuracy : {}'.format(nltk.classify.accuracy(SGDClassifier ,testing_set)*100))
# with open("SGDClassifier.pickle","wb") as f:
#     joblib.dump(SGDClassifier, f)

# print('classification in process for LinearSVC_classifier')
# LinearSVC = SklearnClassifier(LinearSVC())
# LinearSVC.train(training_set)
# print('LinearSVC accuracy : {}'.format(nltk.classify.accuracy(LinearSVC ,testing_set)*100))
# with open("LinearSVC.pickle","wb") as f:
#     joblib.dump(LinearSVC, f)

# print('classification in process for voted_classifier')
# voted_classifier = VoteClassifier(classifier,
#                                   LinearSVC,
#                                   SGDClassifier,
#                                   MNB_classifier,
#                                   LogisticRegression)

# print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

# print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)

