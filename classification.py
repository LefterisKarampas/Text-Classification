from os import path
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn import preprocessing


#Read Train Data
train_data = pd.read_csv('../datasets/train_set.csv', sep="\t")
train_data = train_data[0:400]

test_data = pd.read_csv('../datasets/test_set.csv', sep="\t")
test_data = test_data[0:100]


#Initialize Encoder
le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])

#Initialize CounterVectorizer
count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
X = count_vectorizer.fit_transform(train_data['Content'])
Y = count_vectorizer.transform(test_data['Content'])

#Train classifiers
rclf = RandomForestClassifier()
rclf.fit(X, y)

mclf = MultinomialNB()
mclf.fit(X,y)

sclf = svm.LinearSVC()
sclf.fit(X,y)

#Random_Forest
RFy_pred = rclf.predict(Y)
predicted_categories = le.inverse_transform(RFy_pred)
print(predicted_categories)

#MultinomiaNB
MNBy_pred = mclf.predict(Y)
predicted_categories = le.inverse_transform(MNBy_pred)
print(predicted_categories)

#SVM
SVMy_pred = sclf.predict(Y)
predicted_categories = le.inverse_transform(SVMy_pred)
print(predicted_categories)
