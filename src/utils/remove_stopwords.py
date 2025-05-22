from collections import Counter
from nltk.corpus import stopwords
import nltk
import re
import pandas as pd

stop_words = set(stopwords.words('english'))

#This function removes stop words in a text and return list of words excluding stopwords
def preprocess_text(text: str) -> list:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return words

#This function returns words without stopwords within a dataframe
def removeStopWords(data: pd.DataFrame) -> list:
    words = []
    for text in data['text']:
        words.extend(preprocess_text(text))
    return words

