from os import path
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold,cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import cross_validation
import knn


data = pd.read_csv('../datasets/train_set.csv', sep="\t")
#data = data[0:2000]


le = preprocessing.LabelEncoder()
le.fit(data["Category"])
y = le.transform(data["Category"])



#Initialize CounterVectorizer
#count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
tfid_vectorizer = TfidfVectorizer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False,stop_words=ENGLISH_STOP_WORDS)
#X = count_vectorizer.fit_transform(data['Content']+5*data['Title'])
X = tfid_vectorizer.fit_transform(data['Content']+5*data['Title'])

#LSA - SVD
lsa = TruncatedSVD(n_components=25, n_iter=7, random_state=42)
lsa_X = lsa.fit_transform(X)


#Train classifiers
rclf = RandomForestClassifier()

mclf = MultinomialNB(alpha=0.01)

sclf = svm.SVC(kernel='rbf', C = 10,gamma=1)

myknn = knn.KNN(20)


classifier_pipeline = make_pipeline(preprocessing.StandardScaler(), rclf)
scores = cross_validation.cross_val_score(classifier_pipeline, lsa_X, y, cv=10)
print('Random_Forest: ' + str(scores.mean()))


classifier_pipeline = make_pipeline(preprocessing.MinMaxScaler(feature_range=(50, 100)), mclf)
scores = cross_validation.cross_val_score(classifier_pipeline, lsa_X, y, cv=10)
print('MultinomiaNB: ' + str(scores.mean()))


classifier_pipeline = make_pipeline(preprocessing.StandardScaler(), sclf)
scores = cross_validation.cross_val_score(classifier_pipeline, lsa_X, y, cv=10)
print('SVM: ' + str(scores.mean()))

#classifier_pipeline = make_pipeline(preprocessing.StandardScaler(), myknn)
#scores = cross_validation.cross_val_score(classifier_pipeline, lsa_X, y, cv=10)
#print('KNN: ' + str(scores.mean()))

