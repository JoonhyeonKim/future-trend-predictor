from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

def analyze_trends(text):
    vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
    X = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()
    trend_counter = Counter(keywords)
    trends = trend_counter.most_common(5)
    return trends