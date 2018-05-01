from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold

data = pd.read_csv('../datasets/train_set.csv', sep="\t")

#Initialize Encoder
le = preprocessing.LabelEncoder()
le.fit(data["Category"])
y = le.transform(data["Category"])


#Initialize TfidfVectorizer
tfid_vectorizer = TfidfVectorizer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,stop_words=ENGLISH_STOP_WORDS)

titleWeight = 5
X = tfid_vectorizer.fit_transform(data['Content']+titleWeight*(" "+data['Title']))

#LSA - SVD
lsa = TruncatedSVD(n_components=85, n_iter=7, random_state=42)
lsa_X =lsa.fit_transform(X)

#lsa_X = X


parameters = {'kernel':['linear','rbf'], 'C':range(1,1500,50),'gamma':[0.01, 0.1,1,10]}
grid_search = GridSearchCV(svm.SVC(), parameters, cv=10, n_jobs=-1)
grid_search.fit(lsa_X,y)

print(grid_search.best_params_)

fd = open('SVM_Parameters','w') 
fd.write(str(grid_search.best_params_)+"\n")
fd.close()
