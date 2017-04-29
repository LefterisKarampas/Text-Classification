import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from sklearn.cluster import KMeans
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.preprocessing import StandardScaler




pathToFile = '../train_set.csv'

df = pd.read_csv(pathToFile, sep='\t')
A = np.array(df)

#stop_words = []
stop_words = text.ENGLISH_STOP_WORDS
count_vect = CountVectorizer(stop_words = stop_words)
train_counts = count_vect.fit_transform(df.Content)
svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
svd.fit(train_counts)
km = KMeans(n_clusters=5, random_state=1).fit(svd.transform(train_counts))
#nltk.cluster.KMeansClusterer(num_means=5, distance=cosine_similarity)
#kmeansClusterer.cluster(train_counts, True)
cluster_stats = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
labels = km.labels_.tolist()

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

#the function that finds the distance (metric function) is passed to KmeansClusterer
#kmeansClusterer = nltk.cluster.KMeansClusterer(num_means=5, distance=cosine_similarity)
#kmeansClusterer.cluster(train_counts, True)

#kmeans_mode = Kmeans(n_clusters=5, random_state=0).fit(df.iloc[:, :])
