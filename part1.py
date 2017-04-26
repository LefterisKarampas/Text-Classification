from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

word_string ="word1 word2 word3"

#you can specify fonts, stopwords, background color and other options

wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          width=1200,
                          height=1000
                         ).generate(word_string)


plt.imshow(wordcloud)
plt.axis('off')
plt.show()