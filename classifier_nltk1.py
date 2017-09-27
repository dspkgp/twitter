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

document = [(list(w.lower() for w in movie_reviews.words(fileid) if w not in stop), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

random.shuffle(document)

all_words = [w.lower() for w in movie_reviews.words() if w not in stop]
all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(doc):
    words = set(doc)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# print(find_features(movie_reviews.words('pos/cv012_29576.txt')))
featureset = [(find_features(rev), category) for (rev, category) in document]

#positive data example
training_set = featureset[:1900]
testing_set = featureset[1900:]

#negative data example
# training_set = featureset[100:]
# testing_set = featureset[:100]

print('classification in process for original classifier')
classifier = nltk.NaiveBayesClassifier.train(training_set)
print('original accuracy : {}'.format(nltk.classify.accuracy(classifier ,testing_set)*100))
classifier.show_most_informative_features(15)

import ipdb;ipdb.set_trace()

# print('classification in process for MNB_classifier')
# MNB_classifier = SklearnClassifier(MultinomialNB())
# MNB_classifier.train(training_set)
# print('MNB_classifier accuracy : {}'.format(nltk.classify.accuracy(MNB_classifier ,testing_set)*100))

# print('classification in process for LogisticRegression_classifier')
# LogisticRegression = SklearnClassifier(LogisticRegression())
# LogisticRegression.train(training_set)
# print('LogisticRegression accuracy : {}'.format(nltk.classify.accuracy(LogisticRegression ,testing_set)*100))

# print('classification in process for SGDclassifier')
# SGDClassifier = SklearnClassifier(SGDClassifier())
# SGDClassifier.train(training_set)
# print('SGDClassifier accuracy : {}'.format(nltk.classify.accuracy(SGDClassifier ,testing_set)*100))

# print('classification in process for SVC_classifier')
# SVC = SklearnClassifier(SVC())
# SVC.train(training_set)
# print('SVC accuracy : {}'.format(nltk.classify.accuracy(SVC, testing_set)*100))

# print('classification in process for LinearSVC_classifier')
# LinearSVC = SklearnClassifier(LinearSVC())
# LinearSVC.train(training_set)
# print('LinearSVC accuracy : {}'.format(nltk.classify.accuracy(LinearSVC ,testing_set)*100))

# print('classification in process for voted_classifier')
# voted_classifier = VoteClassifier(classifier,
#                                   LinearSVC,
#                                   SGDClassifier,
#                                   MNB_classifier,
#                                   LogisticRegression)

# print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

# print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)

