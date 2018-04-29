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
import matplotlib.pyplot as plt


data = pd.read_csv('../datasets/train_set.csv', sep="\t")

le = preprocessing.LabelEncoder()
le.fit(data["Category"])
y = le.transform(data["Category"])


#Initialize TfidfVectorizer
tfid_vectorizer = TfidfVectorizer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False,stop_words=ENGLISH_STOP_WORDS)
X = tfid_vectorizer.fit_transform(data['Content']+10*data['Title'])

sclf = svm.SVC(kernel='linear', C = 1000,gamma=0.1)

components = range(10,101,5)

max_value = 0
index = None
accuracy = []
fd = open('lsi_results','w')
#LSA - SVD
for i in components:
	lsa = TruncatedSVD(n_components=i)
	lsa_X = lsa.fit_transform(X)
	lsa_X = preprocessing.StandardScaler().fit_transform(lsa_X)
	scores = cross_validation.cross_val_score(sclf, lsa_X, y, cv=10)
	print(str(i)+": "+str(scores.mean())+"\n")
	fd.write(str(i)+": "+str(scores.mean())+"\n")
	accuracy.append(scores.mean())
	if(scores.mean() > max_value):
		max_value = scores.mean()
		index = i
fd.close()
print('Acurracy: '+str(max_value)+' in components: '+str(index))
plt.plot(components,accuracy)
plt.xlabel("Number of components")
plt.ylabel("Accuracy")
plt.savefig('Components.png')
plt.show()
