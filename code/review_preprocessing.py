# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 14:04:06 2018

@author: granjan
"""
import sys
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize,sent_tokenize, pos_tag
from sklearn.preprocessing import MinMaxScaler 
from nltk.stem.wordnet import WordNetLemmatizer
import string

st = PorterStemmer()
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

input_data = sys.argv[1]
min_sentiment = float(sys.argv[2])
max_sentiment = float(sys.argv[3])
min_wordlen = int(sys.argv[4])
max_wordlen = int(sys.argv[5])
output_file = sys.argv[6]

review_data=pd.read_excel(input_data)

clf = MinMaxScaler(feature_range = (min_sentiment,max_sentiment))

review_data['sentiment_score']=clf.fit_transform(review_data['Sentiment'].reshape(-1,1))

def clean(review_data):
    
    stop_free = " ".join([st.stem(i) for i in review_data.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
    
def processSingleReview(review, d=None):
    """
    Convert a raw review to a string of words
    """
    letters_only = re.sub("[^a-zA-Z]", " ", review)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [st.stem(w) for w in words if w not in stops]
    meaningful_words = [w for w in meaningful_words if pos_tag([w],tagset='universal')[0][1] in ['NOUN','VERB','ADJ']] #
    return(" ".join(meaningful_words))

review_data['clean_sentence'] = review_data.apply(lambda row :clean(row['Sentence']),axis=1)
review_data['clean_sentence']=review_data.apply(lambda row: re.sub(r'\d+', '',row['clean_sentence']),axis=1)

review_data['wordlen'] = review_data.clean_sentence.apply(lambda x: len(x.split()))
review_data = review_data[review_data.wordlen >= min_wordlen]
review_data = review_data[review_data.wordlen <= max_wordlen]

review_data.to_csv(output_file,index=False)
