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


#Read Train Data
data = pd.read_csv('../datasets/train_set.csv', sep="\t")
data = data[0:2000]

train_data ,test_data = train_test_split(data,test_size=0.3)

#Initialize Encoder
le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])
y_test = le.transform(test_data["Category"])


#Initialize CounterVectorizer
#count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
tfid_vectorizer = TfidfVectorizer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False,stop_words=ENGLISH_STOP_WORDS)
#X = count_vectorizer.fit_transform(train_data['Content']+5*train_data['Title'])
X = tfid_vectorizer.fit_transform(train_data['Content']+5*(" "+train_data['Title']))
#Y = count_vectorizer.transform(test_data['Content']+5*test_data['Title'])
Y = tfid_vectorizer.transform(test_data['Content']+5*(" "+test_data['Title']))


#LSA
lsa = TruncatedSVD(n_components=25, n_iter=7, random_state=42)
Scaler = MinMaxScaler(copy=True, feature_range=(50, 100))
lsa_X = lsa.fit_transform(X)
lsa_Y = lsa.fit_transform(Y)
Scaler_X = Scaler.fit_transform(lsa_X)
Scaler_Y = Scaler.fit_transform(lsa_Y)


#Train classifiers
rclf = RandomForestClassifier()
rclf.fit(lsa_X, y)

mclf = MultinomialNB(alpha=0.01)
mclf.fit(Scaler_X,y)

sclf = svm.SVC(kernel='rbf', C = 10,gamma=1)
sclf.fit(lsa_X,y)

myknn = knn.KNN(20)
myknn.fit(lsa_X,y)



#Random_Forest
RFy_pred = rclf.predict(lsa_Y)
predicted_categories = le.inverse_transform(RFy_pred)
print classification_report(y_test, RFy_pred, target_names=list(le.classes_))


#MultinomiaNB
MNBy_pred = mclf.predict(Scaler_Y)
predicted_categories = le.inverse_transform(MNBy_pred)
print classification_report(y_test, MNBy_pred, target_names=list(le.classes_))


#SVM
SVMy_pred = sclf.predict(lsa_Y)
predicted_categories = le.inverse_transform(SVMy_pred)
print classification_report(y_test, SVMy_pred, target_names=list(le.classes_))

#MY_KNN
KNN_pred = myknn.predict(lsa_Y)
predicted_categories = le.inverse_transform(KNN_pred)
print classification_report(y_test, KNN_pred, target_names=list(le.classes_))


