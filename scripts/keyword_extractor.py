from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords(articles):
    vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
    X = vectorizer.fit_transform(articles)
    keywords = vectorizer.get_feature_names_out()
    return keywords
