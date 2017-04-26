import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

stopwords = set(STOPWORDS)
stopwords.add("one")
stopwords.add("two")
stopwords.add("three")
stopwords.add("four")
stopwords.add("five")
stopwords.add("will")
stopwords.add("said")
stopwords.add("year")
stopwords.add("old")
stopwords.add("going")
stopwords.add("see")
stopwords.add("since")
stopwords.add("much")
stopwords.add("now")
stopwords.add("got")
stopwords.add("told")
stopwords.add("given")
stopwords.add("give")
stopwords.add("put")
stopwords.add("made")
stopwords.add("go")
stopwords.add("many")
stopwords.add("However")
stopwords.add("also")
stopwords.add("must")
stopwords.add("week")
stopwords.add("yet")
stopwords.add("may")
stopwords.add("already")
stopwords.add("say")
stopwords.add("still")


df = pd.read_csv('../train_set.csv', sep = '\t')
categories = ['Film']

for category in categories:
	cat = df[df["Category"] == category]
	cat_text = cat[["Content"]]

	word_string = ""
	A = np.array(cat_text)

	for i in range(A.shape[0]):
		for j in range(A.shape[1]):
			word_string += str(A[i,j]) + " "

	wordcloud = WordCloud(stopwords=stopwords,
                          background_color='white',
                          width=1200,
                          height=1000
                         ).generate(word_string)

	plt.imshow(wordcloud)
	plt.axis('off')
	plt.show()