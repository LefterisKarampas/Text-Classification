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
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import knn
from nltk.stem.snowball import SnowballStemmer
import csv

#Read Train Data
train_data = pd.read_csv('../datasets/train_set.csv', sep="\t")
train_data = train_data[0:2000]

test_data = pd.read_csv('../datasets/test_set.csv', sep="\t")
test_data = test_data[0:100]


#Initialize Encoder
le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])


#Stemming
stemmer = SnowballStemmer('english')
analyzer = TfidfVectorizer().build_analyzer()


def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


#tfid_vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words='english', analyzer=ENGLISH_STOP_WORDS, ngram_range=(1,1))
tfid_vectorizer = TfidfVectorizer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False,stop_words=ENGLISH_STOP_WORDS,analyzer=stemmed_words)
#X = count_vectorizer.fit_transform(data['Content']+5*data['Title'])
X = tfid_vectorizer.fit_transform(train_data['Content']+10*(" "+train_data['Title']))
Y = tfid_vectorizer.transform(test_data['Content']+10*(" "+test_data['Title']))

#LSA - SVD
lsa = TruncatedSVD(n_components=25, n_iter=7, random_state=42)
lsa_X = lsa.fit_transform(X)
lsa_Y = lsa.transform(Y)

Scaler = MinMaxScaler(feature_range=(50,100))
Scaler_X = Scaler.fit_transform(lsa_X)
Scaler_Y = Scaler.transform(lsa_Y)

#Train classifiers
rclf = RandomForestClassifier()
rclf.fit(lsa_X, y)

mclf = MultinomialNB()
mclf.fit(Scaler_X,y)

sclf = svm.SVC(kernel='linear', C = 100,gamma=0.1)
sclf.fit(lsa_X,y)

myknn = knn.KNN(50)
myknn.fit(lsa_X,y)

# #Random_Forest
# RFy_pred = rclf.predict(lsa_Y)
# predicted_categories = le.inverse_transform(RFy_pred)
# print(predicted_categories)

# #MultinomiaNB
MNBy_pred = mclf.predict(Scaler_Y)
predicted_categories = le.inverse_transform(MNBy_pred)
#print(predicted_categories)

# #SVM
# SVMy_pred = sclf.predict(lsa_Y)
# predicted_categories = le.inverse_transform(SVMy_pred)
# print(predicted_categories)


# #KNN
# KNNy_pred = myknn.predict(lsa_Y)
# predicted_categories = le.inverse_transform(KNNy_pred)
# print(predicted_categories)

with open('testSet_categories.csv', 'wb') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(['Id','Category'])
    for i in range(len(test_data['Id'])):
      csvwriter.writerow([str(test_data['Id'][i]),predicted_categories[i]])

