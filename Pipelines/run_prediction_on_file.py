from src.predict_text import predict_text
import pandas as pd


PATH = "/home/ec2-user/NewsBucketMount/news_data/trending_news.csv"
save_path = "/home/ec2-user/NewsData/trending_news.csv"

def run_prediction():
    data = pd.read_csv(PATH)
    data = data.dropna(subset=['summary']) 

    data['pred'] = data['summary'].apply(predict_text)

    data.to_csv(save_path)