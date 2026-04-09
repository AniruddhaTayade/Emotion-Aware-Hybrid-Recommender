import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ResourceRecommender:
    def __init__(self, csv_path):
        print("📌 Loading resource dataset...")
        self.df = pd.read_csv(csv_path)

        # Ensure resource_id is int and already 0–based
        self.df["resource_id"] = self.df["resource_id"].astype(int)

        # DO NOT SORT / DO NOT RESET INDEX
        # Index must match resource_id positions exactly

        # Build combined text
        self.df["combined_text"] = (
            self.df["title"].fillna("") + " " + self.df["description"].fillna("")
        )

        print("📌 Building TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["combined_text"])

        print("✅ TF-IDF matrix built:", self.tfidf_matrix.shape)

    def recommend(self, user_text, predicted_emotion, top_k=5, collab=None):
        print(f"\n🔍 Filtering resources for: {predicted_emotion}")

        filtered = self.df[self.df["emotion"] == predicted_emotion]

        if filtered.empty:
            filtered = self.df.copy()

        # resource_id values directly index both matrices
        filtered_ids = filtered["resource_id"].tolist()

        # Content-based similarity
        user_vec = self.vectorizer.transform([user_text])
        filtered_matrix = self.tfidf_matrix[filtered_ids]

        content_scores = cosine_similarity(user_vec, filtered_matrix).flatten()

        # Hybrid
       # HYBRID MODE
        if collab:
            print("🤝 Using Hybrid Recommender")

            collab_scores = collab.predict_scores(filtered_ids)

            # Normalize safely (NumPy 2.0 compatible)
            content_norm = (content_scores - content_scores.min()) / (np.ptp(content_scores) + 1e-8)
            collab_norm = (collab_scores - collab_scores.min()) / (np.ptp(collab_scores) + 1e-8)

            final_scores = 0.7 * content_norm + 0.3 * collab_norm
        else:
            final_scores = content_scores

        # Ranking
        top_idx = np.argsort(final_scores)[::-1][:top_k]
        return filtered.iloc[top_idx][["title", "description", "link", "type", "emotion"]]
