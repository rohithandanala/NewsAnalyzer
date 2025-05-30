import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

#This function is to transform input text using previously saved vectorizer
def process_data(text: str, vectorizer_path: str):
    vectorizer: TfidfVectorizer = joblib.load(vectorizer_path)
    vectorized_input = vectorizer.transform([text])

    return vectorized_input