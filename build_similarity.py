import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import time

print("=" * 55)
print("      AI Movie Recommendation System")
print("=" * 55)

print("\nLoading movies dataset...")
time.sleep(1)

movies = pickle.load(open("movies.pkl", "rb"))

print("✓ Dataset Loaded Successfully")

print("\nGenerating TF-IDF Matrix...")
time.sleep(1)

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(
    movies["Tags"].fillna("")
)

print("✓ TF-IDF Matrix Created")

print("\nGenerating Similarity Matrix...")
time.sleep(1)

similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

print("✓ Similarity Matrix Created")

print("\nSaving Files...")

pickle.dump(tfidf, open("tfidf.pkl", "wb"))
pickle.dump(similarity, open("similarity.pkl", "wb"))

print("\n✓ similarity.pkl Saved")
print("✓ tfidf.pkl Saved")

print("\nProject is Ready 🚀")
print("=" * 55)