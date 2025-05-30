from src.data import process_prediction_data
from joblib import load
import os
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer

def load_model_and_vectorizer(model_name:str):
    
    model_path = f"Models/{model_name}/model.joblib"
    vectorizer_path = f"Models/{model_name}/vectorizer.joblib"


    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Model or vectorizer file not found.")

    model: BaseEstimator = load(model_path)
    vectorizer: TfidfVectorizer = load(vectorizer_path)
    return model, vectorizer


def predict_text(text: str):
    model, vectorizer= load_model_and_vectorizer('logisticRegressionV2')

    # Transform input text
    vectorized_input = vectorizer.transform([text])  # Keep it in list format

    # Predict
    prediction = model.predict(vectorized_input)

    return int(prediction[0]) 
