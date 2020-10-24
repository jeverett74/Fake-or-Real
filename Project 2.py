# import packages (reference madz2000: https://www.kaggle.com/madz2000/nlp-using-glove-embeddings-99-87-accuracy)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from keras.preprocessing import text, sequence
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,plot_confusion_matrix
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet
import keras
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# import datasets
real = pd.read_csv('True.csv')
fake = pd.read_csv('Fake.csv')

real.head()
fake.head()

# Create categories for true or fake news
real['type'] = 1
fake['type'] = 0

real.head()
fake.head()

# the subject values are different, so I will exclude them
real.subject.value_counts()
fake.subject.value_counts()

# combine the datasets
df = pd.concat([real, fake])

df.head()

sns.countplot(df.type)

# check for N/As
df.isna().sum()

# describe dataset
describe = df.describe()
df.shape

# combine title and text into one column and remove other columns
df['text'] = df['title'] + ' ' + df['text']
del df['title']
del df['subject']
del df['date']

df.head()

# set stopwords and punctuation for removal
nltk.download('stopwords')
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

# clean data:
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

# Removing URL's
def remove_between_square_brackets(text):
    return re.sub(r'http\S+', '', text)

#Removing the stopwords from text
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text)
    return text

#Apply function on review column
df['text']=df['text'].apply(denoise_text)

# create a word cloud
# real news
plt.figure()
wc = WordCloud(max_words = 100, stopwords = STOPWORDS).generate(" ".join(df[df.type == 1].text))
plt.imshow(wc , interpolation = 'bilinear')

# fake news
plt.figure()
wc = WordCloud(max_words = 100, stopwords = STOPWORDS).generate(" ".join(df[df.type == 0].text))
plt.imshow(wc , interpolation = 'bilinear')

# how long are real and fake news articles (fake tends to be longer-2500 vs. 5000)
fig,(ax1,ax2)=plt.subplots(1,2)
text_len=df[df['type']==1]['text'].str.len()
ax1.hist(text_len)
ax1.set_title('Real News')
text_len=df[df['type']==0]['text'].str.len()
ax2.hist(text_len)
ax2.set_title('Fake News')
fig.suptitle('Length of Articles')
plt.show()

# word lengths (fake tends to have shorter average words, but outliers with bigger words)
fig,(ax1,ax2)=plt.subplots(1,2)
word=df[df['type']==1]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1)
ax1.set_title('Real News')
word=df[df['type']==0]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2)
ax2.set_title('Fake News')
fig.suptitle('Average word lengths')

# convert all words to list
def get_words(text):
    words = []
    for i in text:
        for j in i.split():
            words.append(j.strip())
    return words
words = get_words(df.text)
words[:5]

# count most common words
word_count = Counter(words)
most = word_count.most_common(10)
most = dict(most)
most


# define ngram formula
def get_top_text_ngrams(corpus, n, g):
    vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

# unigrams
plt.figure()
unigram = get_top_text_ngrams(df.text,10,1)
unigram = dict(unigram)
sns.barplot(x=list(unigram.values()),y=list(unigram.keys()))

# bigrams
plt.figure()
bigram = get_top_text_ngrams(df.text,10,2)
bigram = dict(bigram)
sns.barplot(x=list(bigram.values()),y=list(bigram.keys()))

# trigrams
plt.figure(figsize=(20,10))
trigram = get_top_text_ngrams(df.text,10,3)
trigram = dict(trigram)
sns.barplot(x=list(trigram.values()),y=list(trigram.keys()))

# split data into test and train
x_train,x_test,y_train,y_test = train_test_split(df.text,df.type,random_state = 0)

pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression())])

# fit model
model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)


# confusion matrix and accuracy
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))

plot_confusion_matrix(model,x_test,y_test)