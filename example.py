import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
 
# The kmeans algorithm is implemented in the scikits-learn library
from sklearn.cluster import KMeans
 
 
pathToFile = '/home/lef/Desktop/train_set.csv'
 
df = pd.read_csv(pathToFile, sep='\t')
stop_words=['Antonia','Nikos','Nikolas']
count_vect = CountVectorizer(stop_words = stop_words)
svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
count_vect.fit(df.Content)
train_counts = count_vect.transform(df.Content)
	# Create a kmeans model on our data, using k clusters.  random_state helps ensure that the algorithm returns the same results each time.
kmeans_model = KMeans(n_clusters=5, random_state=1).svd.fit(df.Content)
	
	# These are our fitted labels for clusters -- the first cluster has label 0, and the second has label 1.
labels = kmeans_model.labels_
 
	# Sum of distances of samples to their closest cluster center
interia = kmeans_model.inertia_
print ("k:",k, " cost:", interia)
