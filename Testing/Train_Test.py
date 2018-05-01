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

import sys
sys.path.insert(0,'../KNN/')

import knn


stopwords = set(ENGLISH_STOP_WORDS)

stopwords.add("yes")
stopwords.add("no")
stopwords.add("know")
stopwords.add("also")
stopwords.add("told")
stopwords.add("one")
stopwords.add("two")
stopwords.add("first")
stopwords.add("last")
stopwords.add("new")
stopwords.add("say")
stopwords.add("year")
stopwords.add("thing")
stopwords.add("something")
stopwords.add("now")
stopwords.add("said")
stopwords.add("even")
stopwords.add("will")
stopwords.add("although")
stopwords.add("always")
stopwords.add("often")
stopwords.add("day")
stopwords.add("us")
stopwords.add("years")
stopwords.add("another")
stopwords.add("came")

#Read Train Data
data = pd.read_csv('../datasets/train_set.csv', sep="\t")

train_data ,test_data = train_test_split(data,test_size=0.3)

#Initialize Encoder
le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])
y_test = le.transform(test_data["Category"])


#Initialize TfidfVectorizer
tfid_vectorizer = TfidfVectorizer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,stop_words=stopwords)
X = tfid_vectorizer.fit_transform(train_data['Content']+5*(" "+train_data['Title']))
Y = tfid_vectorizer.transform(test_data['Content']+5*(" "+test_data['Title']))


#LSA
lsa = TruncatedSVD(n_components=25, n_iter=7, random_state=42)
lsa_X = lsa.fit_transform(X)
lsa_Y = lsa.transform(Y)

#lsa_X =X
#lsa_Y =Y

for i in range(5,100,5):	
	print(str(i)+": ")
	sclf = knn.KNN(i)
	#sclf = svm.SVC(kernel='linear', C = 1000,gamma=0.01)
	sclf.fit(lsa_X,y)


	#SVM
	SVMy_pred = sclf.predict(lsa_Y)
	predicted_categories = le.inverse_transform(SVMy_pred)
	print(classification_report(y_test, SVMy_pred, target_names=list(le.classes_)))


print("Finish")