import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def process_data(data: pd.DataFrame):
    print('preparing data for traning the model')
    x = data["text"]
    Y = data["class"]

    x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.2, random_state=42)

    #Initializing Vectorizer to convert text into numbers for the machine to understand
    vectorizer = TfidfVectorizer()

    xv_train = vectorizer.fit_transform(x_train)
    xv_test = vectorizer.transform(x_test)  


    return xv_train, xv_test, Y_train, Y_test, vectorizer