# import requests
# import os
# from dotenv import load_dotenv
# from datetime import datetime, timedelta

# # Load environment variables from the .env file
# load_dotenv()

# def fetch_news(days=7):
#     api_key = os.getenv("NEWS_API_KEY")
#     end_date = datetime.now()
#     start_date = end_date - timedelta(days=days)
    
#     url = f"https://newsapi.org/v2/everything?q=technology&from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&sortBy=publishedAt&apiKey={api_key}"
    
#     response = requests.get(url)
#     if response.status_code == 200:
#         data = response.json()
#         articles = data.get("articles", [])
#         return [article["content"] for article in articles if article.get("content")]
#     else:
#         print(f"Failed to fetch news: {response.status_code}")
#         return []
from langchain_teddynote.tools import GoogleNews
from typing import List, Dict

def fetch_news(days=7, query="기술") -> List[str]:
    news_tool = GoogleNews()
    articles = news_tool.search_by_keyword(query, k=100)  # Fetch up to 100 articles
    content_list = [article['content'] for article in articles if 'content' in article]
    
    if not content_list:
        print("No news articles found.")
    
    return content_list
