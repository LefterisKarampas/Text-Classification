from os import path
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold,cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import cross_validation
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt


data = pd.read_csv('../datasets/train_set.csv', sep="\t")

le = preprocessing.LabelEncoder()
le.fit(data["Category"])
y = le.transform(data["Category"])


#Initialize TfidfVectorizer
tfid_vectorizer = TfidfVectorizer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False,stop_words=ENGLISH_STOP_WORDS)
X = tfid_vectorizer.fit_transform(data['Content']+10*(" "+data['Title']))

sclf = svm.SVC(kernel='linear', C = 1000,gamma=0.1)

components = range(25,150,5)

max_value = 0
index = None
accuracy = []
fd = open('lsi_results','w')
#LSA - SVD
for i in components:
	lsa = TruncatedSVD(n_components=i,n_iter=7, random_state=42)
	lsa_X = lsa.fit_transform(X)
	#lsa_X = preprocessing.StandardScaler().fit_transform(lsa_X)
	scores = cross_validate(sclf, lsa_X, y, cv=10,scoring=['accuracy'])
	mean_value = np.mean(scores['test_accuracy'])
	print(str(i)+": "+str(mean_value)+"\n")
	fd.write(str(i)+": "+str(mean_value)+"\n")
	accuracy.append(mean_value)
	if(mean_value > max_value):
		max_value = mean_value
		index = i
fd.close()
print('Acurracy: '+str(max_value)+' in components: '+str(index))
plt.plot(components,accuracy)
plt.xlabel("Number of components")
plt.ylabel("Accuracy")
plt.savefig('Components.png')
#plt.show()
