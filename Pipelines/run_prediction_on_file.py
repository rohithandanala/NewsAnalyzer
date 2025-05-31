from src import predict_text
import pandas as pd


PATH = "../news_data/trending_news.csv"

def run_prediction():
    data = pd.read_csv(PATH)

    data['pred'] = data['title'].apply(predict_text)

    data.to_csv(PATH)