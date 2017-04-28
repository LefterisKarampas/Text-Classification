import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk

pathToFile = '/home/mt/Desktop/DI/DataMining/prj1/train_set.csv'

df = pd.read_csv(pathToFile, sep='\t')
Array = np.array(df)

#the function that finds the distance (metric function) is passed to KmeansClusterer
kmeansClusterer = nltk.cluster.KMeansClusterer(num_means=5, distance=cosine_similarity)
kmeansClusterer.cluster(Array)
