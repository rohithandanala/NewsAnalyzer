import requests
import pandas as pd
import boto3
import io
import json

NEWS_API_BEARER_TOKEN = "pub_77556b8744f45973a75717b8ce91d38271a69"
BUCKET_NAME = "my-news-application"
CSV_KEY = "news_data/trending_news.csv"  # You can customize this path

def get_us_english_trends_df():
    url = f"https://newsdata.io/api/1/news?apikey={NEWS_API_BEARER_TOKEN}&country=us&language=en&category=top"
    
    response = requests.get(url)
    if response.status_code != 200:
        print("Error fetching news:", response.json())
        return pd.DataFrame()
    
    data = response.json()
    
    news_list = []
    for article in data.get("results", []):
        news_list.append({
            "id": article.get("article_id"),
            "title": article.get("title"),
            "summary": article.get("description", "No summary available"),
            "image": article.get("image_url", ""),
            "link": article.get("link"),
            "published_at": article.get("pubDate")
        })
    
    return pd.DataFrame(news_list)

def lambda_handler(event=None, context=None):
    try:
        df = get_us_english_trends_df()
        if df.empty:
            return {
                "statusCode": 500,
                "body": json.dumps("No news data fetched.")
            }

        # Convert DataFrame to CSV in memory
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)

        # Upload to S3
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=CSV_KEY,
            Body=csv_buffer.getvalue()
        )

        return {
            "statusCode": 200,
            "body": json.dumps({"message": "News data fetched and stored to S3 successfully"})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(str(e))
        }
