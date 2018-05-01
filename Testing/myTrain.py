from os import path
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.cross_validation import train_test_split
import csv
from sklearn.feature_selection import SelectKBest, chi2
from nltk import stem
from nltk.corpus import stopwords
from preprocessing import *

import sys
sys.path.insert(0,'../KNN/')

import knn

#Read Train Data
#data = pd.read_csv('../datasets/train_set.csv', sep="\t")
#data = data[0:2000]

train_data = pd.read_csv('../datasets/train_set.csv', sep="\t")
#train_data = train_data[0:2000]

test_data = pd.read_csv('../datasets/test_set.csv', sep="\t")

#train_data ,test_data = train_test_split(data,test_size=0.3)

#Initialize Encoder
le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])
#y_test = le.transform(test_data["Category"])

titleWeight = 5
train_text = train_data['Content']+titleWeight*(" " + train_data['Title'])
text_text = test_data['Content']+titleWeight*(" " + test_data['Title'])

train_X, test_X = stem_and_tfidf(train_text, text_text)
# features_num = 2000
# tX = train_X
# tY = test_X
# for features_num in range(2000,10000,500):
# 	print(features_num)
# 	train_X, test_X = select_features(tX, y, tY, features_num)

#SVM
sclf = svm.SVC(kernel='linear', C = 301,gamma=0.01)
sclf.fit(train_X,y)
SVMy_pred = sclf.predict(test_X)
predicted_categories = le.inverse_transform(SVMy_pred)
#print classification_report(y_test, SVMy_pred, target_names=list(le.classes_))

with open('testSet_categories.csv', 'wb') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(['Id','Category'])
    for i in range(len(test_data['Id'])):
      csvwriter.writerow([str(test_data['Id'][i]),predicted_categories[i]])