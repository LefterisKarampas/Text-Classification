
from os import path
from wordcloud import WordCloud,STOPWORDS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train_data = pd.read_csv('../datasets/train_set.csv', sep = '\t')
categories = ['Politics', 'Film', 'Football', 'Business', 'Technology']

stopwords = set(STOPWORDS)

train_data = train_data[0:25]

string = {'Politics':"",'Film':"",'Football':"",'Business':"",'Technology':""}

for i in range(len(train_data)):
	string[train_data.iloc[i]["Category"]] += train_data.iloc[i]["Content"] + train_data.iloc[i]["Title"]
	

for category in categories:
	if(len(string[category]) > 0):
		wordcloud = WordCloud(stopwords=stopwords,
	                          background_color='black',
	                          width=1800,
	                          height=1200,
	                          collocations = False
	                         ).generate(string[category])
		plt.imshow(wordcloud)
		plt.axis('off')
		plt.show()
