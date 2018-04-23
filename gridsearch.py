from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing


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

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 100],'gamma':[0.001, 0.01, 0.1, 1]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters,cv=10)
clf.fit(lsa_X,y)

print(clf.best_params_)
