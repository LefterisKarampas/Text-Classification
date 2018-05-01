from os import path
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
import csv

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
train_data = pd.read_csv('../datasets/train_set.csv', sep="\t")

#Read Test_Data
test_data = pd.read_csv('../datasets/test_set.csv', sep="\t")

#Initialize Encoder
le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])


#Stemming
stemmer = SnowballStemmer('english')
analyzer = TfidfVectorizer().build_analyzer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


tfid_vectorizer = TfidfVectorizer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False,stop_words=stopwords,analyzer=stemmed_words)

titleWeight = 5
X = tfid_vectorizer.fit_transform(train_data['Content']+titleWeight*(" "+train_data['Title']))
Y = tfid_vectorizer.transform(test_data['Content']+titleWeight*(" "+test_data['Title']))

#LSA - SVD
#lsa = TruncatedSVD(n_components=85, n_iter=7, random_state=42)
#lsa_X = lsa.fit_transform(X)
#lsa_Y = lsa.transform(Y)

lsa_X = X
lsa_Y = Y

#Initialize classifier and train_data
#sclf = knn.KNN(5)
sclf = svm.SVC(kernel='linear', C = 1000,gamma=0.01)
sclf.fit(lsa_X,y)

#Predict test_data
SVMy_pred = sclf.predict(lsa_Y)
predicted_categories = le.inverse_transform(SVMy_pred)
print(predicted_categories)


with open('testSet_categories.csv', 'wb') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(['Id','Category'])
    for i in range(len(test_data['Id'])):
      csvwriter.writerow([str(test_data['Id'][i]),predicted_categories[i]])

