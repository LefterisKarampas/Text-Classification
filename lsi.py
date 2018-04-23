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

sclf = svm.SVC(kernel='rbf', C = 10,gamma=1)

components = range(1,100)


max_value = 0
index = None
#LSA - SVD
for i in range(1,100):
	lsa = TruncatedSVD(n_components=i, n_iter=7, random_state=42)
	lsa_X = lsa.fit_transform(X)
	classifier_pipeline = make_pipeline(preprocessing.StandardScaler(), sclf)
	scores = cross_validation.cross_val_score(classifier_pipeline, lsa_X, y, cv=10)
	if(scores.mean() > max_value):
		max_value = scores.mean()
		index = i

print('Acurracy: '+str(max_value)+' in components: '+str(index))