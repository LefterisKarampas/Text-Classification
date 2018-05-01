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
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold,cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from nltk.stem.snowball import SnowballStemmer
import numpy as np
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

data = pd.read_csv('../datasets/train_set.csv', sep="\t")

#Initialize Encoder
le = preprocessing.LabelEncoder()
le.fit(data["Category"])
y = le.transform(data["Category"])


titleWeight = 5
X = data['Content']+titleWeight*(" "+data['Title'])

#Initialize Transformer
#Stemming
stemmer = SnowballStemmer('english')
analyzer = TfidfVectorizer().build_analyzer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


tfid_vectorizer = TfidfVectorizer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,stop_words=stopwords,analyzer=stemmed_words)
X = tfid_vectorizer.fit_transform(X)

#LSA - SVD
lsa = TruncatedSVD(n_components=85, n_iter=7, random_state=42)
lsa_X = lsa.fit_transform(X)


#Train classifiers

sclf = svm.SVC(kernel='linear', C = 1000,gamma=0.01)


csv_out = [['Statistic Measure','My Method'],
		['Accuracy'],
		['Precision'],
		['Recall'],
		['F-Measure']]

scoring = ['accuracy','precision_macro', 'recall_macro','f1_macro']

classifiers = [sclf]

for classifier in classifiers:
	classifier_pipeline = make_pipeline(classifier)
	scores = cross_validate(classifier_pipeline, lsa_X, y, cv=10,scoring=scoring)
	csv_out[1].append(np.mean(scores['test_accuracy']))
	csv_out[2].append(np.mean(scores['test_precision_macro']))
	csv_out[3].append(np.mean(scores['test_recall_macro']))
	csv_out[4].append(np.mean(scores['test_f1_macro']))



fd = open('MyMethod.csv','w')
for i in range(len(csv_out)):
	for j in range(len(csv_out[i])):
		if(i > 0 and j > 0):
			fd.write(str(round(csv_out[i][j],5))+"\t")
		else:
			fd.write(str(csv_out[i][j])+"\t")
	fd.write("\n")

fd.close()