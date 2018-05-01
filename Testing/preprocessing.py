import re
from time import time
import argparse
from collections import Counter

import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from nltk import stem
from nltk.corpus import stopwords
from nltk import download

download('stopwords')
download('wordnet')

class StemTokenizer(object):
    def __init__(self):
        self.wnl = stem.WordNetLemmatizer()
        self.word = re.compile('[a-zA-Z]+')

    def __call__(self, doc):
        tokens = re.split('\W+', doc.lower())
        tokens = [self.wnl.lemmatize(t) for t in tokens]
        return tokens

def stem_and_tfidf(train_data, test_data):
    stops = stopwords.words('english')
    counter = CountVectorizer(tokenizer=StemTokenizer(),
                              stop_words=stops, min_df=3,
                              dtype=np.double)
    counter.fit(train_data)
    train_tf = counter.transform(train_data)
    test_tf = counter.transform(test_data)

    transformer = TfidfTransformer(norm='l2', sublinear_tf=True, use_idf=True)
    train_tfidf = transformer.fit_transform(train_tf)
    test_tfidf = transformer.transform(test_tf)
    return train_tfidf, test_tfidf

def select_features(train_X, train_y, test_X, k):
    selector = SelectKBest(chi2, k=k)
    selector.fit(train_X, train_y)
    train_X = selector.transform(train_X)
    test_X = selector.transform(test_X)
    return train_X, test_X
