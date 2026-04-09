import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

class CollaborativeRecommender:
    def __init__(self, ratings_csv, num_components=20):
        print("📌 Loading rating dataset...")
        df = pd.read_csv(ratings_csv)

        # No index shifting – your CSV already uses 0–114 correctly
        df["resource_id"] = df["resource_id"].astype(int)

        # Pivot
        self.user_item = df.pivot_table(
            index="user_id",
            columns="resource_id",
            values="rating",
            fill_value=0
        )

        print("📌 Running SVD...")
        svd = TruncatedSVD(n_components=num_components)
        self.latent_matrix = svd.fit_transform(self.user_item)

        self.resource_factors = svd.components_.T

    def predict_scores(self, resource_ids):
        factors = self.resource_factors[resource_ids]
        center_vec = factors.mean(axis=0)
        return factors.dot(center_vec)
