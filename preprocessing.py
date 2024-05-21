import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
import re
from nltk.tokenize import WhitespaceTokenizer
import nltk
from collections import Counter
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('spam.csv', encoding="latin-1")
 
# shuffling all our data
df = df.sample(frac=1)

# drop unnecessary columns
df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
# rename features
df.rename(columns={"v1": "labels", "v2": "message"}, inplace=True)

df['message']= df['message'].apply(lambda x: x.lower())

#defining the function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree
 
# storing the punctuation free text in data frame
df['message']= df['message'].apply(lambda x:remove_punctuation(x))

# tokenize text 
def tokenization(text):
    tk = WhitespaceTokenizer()
    return tk.tokenize(text)
 
# applying function to the column for making tokens
df['message']= df['message'].apply(lambda x: tokenization(x))

nltk.download('stopwords')
# Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')
 
#defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output
 
#applying the function for removal of stopwords
df['message']= df['message'].apply(lambda x:remove_stopwords(x))

# count words
cnt = Counter()
for text in df["message"].values:
    for word in text:
        cnt[word] += 1

freq_words = set([w for (w, wc) in cnt.most_common(10)])
rare_words = set([w for w in cnt if cnt[w] < 3])


def remove_words(text, freq):
    """custom function to remove the frequent words"""
    return [word for word in text if word not in freq]


# remove 10 most common words
df['message']= df['message'].apply(lambda x:remove_words(x, freq_words))
# remove words that appear only one or two times
df['message']= df['message'].apply(lambda x:remove_words(x, rare_words))

#defining the object for stemming
porter_stemmer = PorterStemmer()
 
#defining a function for stemming
def stemming(text):
  stem_text = [porter_stemmer.stem(word) for word in text]
  return stem_text
 
# applying function for stemming
df['message']=df['message'].apply(lambda x: stemming(x))

# trasform the list of words in messages
df['message']=df['message'].apply(lambda x: ' '.join(x))

# split training and test set
x_train,x_test,y_train,y_test = train_test_split(df["message"],df["labels"],test_size=0.3)





# vectorize words
def vectorize(data,tfidf_vect_fit):
    X_tfidf = tfidf_vect_fit.transform(data)
    words = tfidf_vect_fit.get_feature_names_out()
    X_tfidf_df = pd.DataFrame(X_tfidf.toarray())
    X_tfidf_df.columns = words
    return(X_tfidf_df)

tfidf_vect = TfidfVectorizer()
tfidf_vect_fit=tfidf_vect.fit(x_train)
x_train=vectorize(x_train,tfidf_vect_fit)
#tfidf_vect_fit=tfidf_vect.fit(x_test)
x_test=vectorize(x_test,tfidf_vect_fit)


x_train.to_csv('x_train_set.csv', index=False)
y_train.to_csv('y_train_set.csv', index=False)
x_test.to_csv('x_test_set.csv', index=False)
y_test.to_csv('y_test_set.csv', index=False)
