import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import nltk

pathToFile = '/home/mt/Desktop/DI/DataMining/prj1/train_set.csv'

df = pd.read_csv(pathToFile, sep='\t')
stop_words=['Antonia','Nikos','Nikolas']
count_vect = CountVectorizer(stop_words = stop_words)
count_vect.fit(df.Content)
train_counts = count_vect.transform(df.Content)
#train_counts.shape
print (train_counts)

#the function that finds the distance (metric function) is passed to KmeansClusterer
#kmeansClusterer = nltk.cluster.KMeansClusterer(num_means=5, distance=cosine_similarity)
#kmeansClusterer.cluster(train_counts)
