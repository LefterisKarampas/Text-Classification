import pandas as pd
import numpy as np
import nltk
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.feature_extraction import text
from sklearn.cluster import KMeans
from sklearn.decomposition import RandomizedPCA

pathToFile = '../train_set.csv'

df = pd.read_csv(pathToFile, sep='\t')
A = np.array(df.Content)
stop_words = text.ENGLISH_STOP_WORDS

cv = CountVectorizer(stop_words = stop_words)
svd = TruncatedSVD(n_components=2, n_iter=5, random_state=42)

c_vector = cv.fit_transform(df.Content.iloc[0:100])
reduced_vector = svd.fit_transform(c_vector)
l_of_lists = reduced_vector.tolist()
#print (l_of_lists)
# for f in l_of_lists:
# 	print (f)
np_vector = [np.array([f]) for f in l_of_lists]

print ("Preprocessing Done!Starting kmeans")
km = nltk.cluster.KMeansClusterer(num_means=5, distance=cosine_similarity, avoid_empty_clusters=True)
clusters = km.cluster(np_vector)
print (clusters)
cluster_stats = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
labels = clusters

for i,label in enumerate(labels):
	if (A[i][4] == "Politics"):
		cluster_stats[label][0] += 1
		cluster_stats[label][5] += 1
	elif (A[i][4] == "Film"):
		cluster_stats[label][1] += 1
		cluster_stats[label][5] += 1
	elif (A[i][4] == "Football"):
		cluster_stats[label][2] += 1
		cluster_stats[label][5] += 1
	elif (A[i][4] == "Business"):
		cluster_stats[label][3] += 1
		cluster_stats[label][5] += 1
	else: #Technology
		cluster_stats[label][4] += 1
		cluster_stats[label][5] += 1

categories = ['Politics\t', 'Film\t\t', 'Football\t', 'Business\t', 'Technology\t']

for i in range(5):
	print("Cluster {0:d}:".format(i+1))
	for j,cat in enumerate(categories):
		print ('\t', cat, ": {0:.2f}%".format(cluster_stats[i][j] *100 / cluster_stats[i][5]))
	print('\n')