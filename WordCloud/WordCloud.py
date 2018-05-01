import os
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread


train_data = pd.read_csv('../datasets/train_set.csv', sep = '\t')
categories = ['Politics', 'Film', 'Football', 'Business', 'Technology']

stopwords = set(STOPWORDS)

stopwords.add("yes")
stopwords.add("no")
stopwords.add("know")
stopwords.add("also")
stopwords.add("told")
stopwords.add("one")
stopwords.add("two")
stopwords.add("first")
stopwords.add("last")
stopwords.add("new")
stopwords.add("say")
stopwords.add("year")
stopwords.add("thing")
stopwords.add("something")
stopwords.add("now")
stopwords.add("said")
stopwords.add("even")
stopwords.add("will")
stopwords.add("although")
stopwords.add("always")
stopwords.add("often")
stopwords.add("day")
stopwords.add("us")
stopwords.add("years")
stopwords.add("another")
stopwords.add("came")

colors = imread("coloring.jpg")
image_colors = ImageColorGenerator(colors)

string = {'Politics':"",'Film':"",'Football':"",'Business':"",'Technology':""}

directory = "wordcloud"
if not os.path.exists(directory):
    os.makedirs(directory)

for i in range(len(train_data)):
	string[train_data.iloc[i]["Category"]] += train_data.iloc[i]["Content"] + " "
	
for category in categories:
	if(len(string[category]) > 0):
		wordcloud = WordCloud(stopwords=stopwords,
	                          background_color='white',
	                          width=1920,
	                          height=1080,
	                          collocations = False,
	                          max_words = 500
	                         ).generate(string[category])
		plt.imshow(wordcloud.recolor(color_func=image_colors))
		wordcloud.to_file("wordcloud/"+category+".png")
