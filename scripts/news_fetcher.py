import requests
import os
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

def fetch_news():
    api_key = os.getenv("NEWS_API_KEY")
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        articles = data.get("articles", [])
        return [article["content"] for article in articles if article.get("content")]
    else:
        print(f"Failed to fetch news: {response.status_code}")
        return []