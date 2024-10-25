import json
from chromadb import Client, Settings

def index_news(articles):
    client = Client(Settings(persist_directory="./data/chroma_db"))
    collection = client.create_collection("news_articles")

    for i, article in enumerate(articles):
        collection.add(
            documents=[article],
            metadatas=[{"source": f"article_{i}"}],
            ids=[f"id_{i}"]
        )

    # Save articles to a JSON file for backup
    with open("./data/news_database.json", "w") as f:
        json.dump(articles, f)
