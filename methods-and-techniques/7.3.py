documents = [
    "I love programming in Python",
    "Java is a powerful language.",
    "Python is great for data science.",
    "I prefer Java for system programming",
    "Machine Learning is fun with Python"
]

# example of bag of words (count vectorizer)
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()
X_count = count_vectorizer.fit_transform(documents)
print("Count Vectorizer:\n", X_count.toarray())
print(count_vectorizer.get_feature_names_out())

# example of TF-IDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(documents)
print("TF-IDF:\n", X_tfidf.toarray())
print(tfidf_vectorizer.get_feature_names_out())