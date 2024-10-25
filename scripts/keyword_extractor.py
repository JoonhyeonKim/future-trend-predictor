# from sklearn.feature_extraction.text import TfidfVectorizer

# def extract_keywords(articles):
#     vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
#     X = vectorizer.fit_transform(articles)
#     keywords = vectorizer.get_feature_names_out()
#     return keywords
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt

def extract_keywords(articles):
    okt = Okt()
    tokenized_articles = [' '.join(okt.nouns(article)) for article in articles if isinstance(article, str)]
    
    if not tokenized_articles:
        print("No valid articles found for keyword extraction.")
        return []

    vectorizer = TfidfVectorizer(max_features=10)
    X = vectorizer.fit_transform(tokenized_articles)
    keywords = vectorizer.get_feature_names_out().tolist()  # Convert to list
    return keywords
