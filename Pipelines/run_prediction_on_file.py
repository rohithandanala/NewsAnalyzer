from src.predict_text import predict_text
import pandas as pd


def run_prediction(PATH, save_path):
    data = pd.read_csv(PATH)
    data = data.dropna(subset=['summary']) 

    data['pred'] = data['summary'].apply(predict_text)

    data.to_csv(save_path)