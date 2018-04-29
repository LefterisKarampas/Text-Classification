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
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
import numpy as np
import knn


data = pd.read_csv('../datasets/train_set.csv', sep="\t")



le = preprocessing.LabelEncoder()
le.fit(data["Category"])
y = le.transform(data["Category"])

X = data['Content']+10*(" "+data['Title'])
#Initialize CounterVectorizer
#count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
#X = count_vectorizer.fit_transform(data['Content']+10*(" "+data['Title']))

#Initialize Transformer
tfid_vectorizer = TfidfVectorizer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
X = tfid_vectorizer.fit_transform(X)

#LSA - SVD
lsa = TruncatedSVD(n_components=25, n_iter=7, random_state=42)
lsa_X = lsa.fit_transform(X)


#Train classifiers
rclf = RandomForestClassifier()

mclf = MultinomialNB(alpha=0.01)

sclf = svm.SVC(kernel='linear', C = 1000,gamma=0.1)

myknn = knn.KNN(20)


csv_out = [['Statistic Measure','Naive Bayes','Random_Forest','SVM','KNN','My Method'],
		['Accuracy'],
		['Precision'],
		['Recall'],
		['F-Measure']]

scoring = ['accuracy','precision_macro', 'recall_macro','f1_macro']

classifiers = [rclf,mclf,sclf,myknn]

for classifier in classifiers:
	if(classifier == mclf):
		classifier_pipeline = make_pipeline(preprocessing.MinMaxScaler(feature_range=(50, 100)), classifier)
	else:	
		classifier_pipeline = make_pipeline(preprocessing.StandardScaler(), classifier)
	scores = cross_validate(classifier_pipeline, lsa_X, y, cv=10,scoring=scoring)
	csv_out[1].append(np.mean(scores['test_accuracy']))
	csv_out[2].append(np.mean(scores['test_precision_macro']))
	csv_out[3].append(np.mean(scores['test_recall_macro']))
	csv_out[4].append(np.mean(scores['test_f1_macro']))



fd = open('EvaluationMetric_10fold.csv','w')
for i in range(len(csv_out)):
	for j in range(len(csv_out[i])):
		if(i > 0 and j > 0):
			fd.write(str(round(csv_out[i][j],5))+"\t")
		else:
			fd.write(str(csv_out[i][j])+"\t")
	fd.write("\n")

fd.close()