import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from sklearn.cluster import KMeans

pathToFile = '../train_set.csv'

df = pd.read_csv(pathToFile, sep='\t')
A = np.array(df)

stop_words=['Antonia','Nikos','Nikolas']
count_vect = CountVectorizer(stop_words = stop_words)
train_counts = count_vect.fit_transform(df.Content)
svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
#train_counts.shape
print (train_counts)
X = svd.fit(train_counts)
km = KMeans(n_clusters=5, random_state=1).fit(svd.transform(train_counts))

cluster_stats = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
labels = km.labels_.tolist()

for i,label in enumerate(labels):
	if (A[i][3] is "Politics"):
		cluster_stats[label][0] += 1
		cluster_stats[label][5] += 1
	elif (A[i][3] is "Film"):
		cluster_stats[label][1] += 1
		cluster_stats[label][5] += 1
	elif (A[i][3] is "Footbal"):
		cluster_stats[label][2] += 1
		cluster_stats[label][5] += 1
	elif (A[i][3] is "Business"):
		cluster_stats[label][3] += 1
		cluster_stats[label][5] += 1
	else: #Technology
		cluster_stats[label][4] += 1
		cluster_stats[label][5] += 1

categories = ['Politics', 'Film', 'Football', 'Business', 'Technology']

for i in range(5):
	print ("Cluster ", i)
	for j,cat in enumerate(categories):
		print ('\t', cat, ': ', cluster_stats[i][j] / cluster_stats[i][5], '%')

#the function that finds the distance (metric function) is passed to KmeansClusterer
#kmeansClusterer = nltk.cluster.KMeansClusterer(num_means=5, distance=cosine_similarity)
#kmeansClusterer.cluster(train_counts, True)

#kmeans_mode = Kmeans(n_clusters=5, random_state=0).fit(df.iloc[:, :])
