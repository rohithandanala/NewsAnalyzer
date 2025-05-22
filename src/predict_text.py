from src.data import process_prediction_data
from joblib import load


def predict_text(text: str):
    # Load vectorizer
    vectorizer = load("Models/vectorizer.pkl")

    # Load trained model
    model = load(f"Models/{model_name}.joblib")

    # Transform input text
    vectorized_input = vectorizer.transform([text])  # Keep it in list format

    # Predict
    prediction = model.predict(vectorized_input)

    return prediction[0]
