import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

import joblib


def train_logreg_baseline():

    print("📌 Loading dataset...")
    df = pd.read_csv("data/cleaned_goemotions.csv")

    X = df["text"]
    y = df["emotion"]

    # ------------------------------
    # TF-IDF VECTORIZATION
    # ------------------------------
    print("🔤 Building TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=8000,
        stop_words="english",
        ngram_range=(1,2)   # slightly better performance
    )
    X_vec = vectorizer.fit_transform(X)

    # ------------------------------
    # TRAIN / TEST SPLIT
    # ------------------------------
    print("📚 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------------------
    # LOGISTIC REGRESSION MODEL
    # ------------------------------
    print("🤖 Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",    # handles class imbalance
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # ------------------------------
    # EVALUATION
    # ------------------------------
    print("🔍 Evaluating model...")
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="weighted"
    )

    print("\n==============================")
    print("📊 LOGISTIC REGRESSION RESULTS")
    print("==============================")
    print(f"Accuracy :  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\n📝 Classification Report:")
    print(classification_report(y_test, preds))

    # ------------------------------
    # CONFUSION MATRIX
    # ------------------------------
    print("\n🖼 Generating Confusion Matrix...")

    labels = sorted(df["emotion"].unique())
    cm = confusion_matrix(y_test, preds, labels=labels)

    plt.figure(figsize=(18, 14))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted Emotion")
    plt.ylabel("True Emotion")
    plt.title("Confusion Matrix - Logistic Regression Baseline")
    plt.tight_layout()

    # Save image
    plt.savefig("models/logreg_confusion_matrix.png")
    plt.close()

    print("✅ Confusion Matrix saved → models/logreg_confusion_matrix.png")

    # ------------------------------
    # SAVE MODEL + VECTORIZER
    # ------------------------------
    joblib.dump(model, "models/logreg_emotion_model.pkl")
    joblib.dump(vectorizer, "models/logreg_vectorizer.pkl")

    print("💾 Model + Vectorizer saved to models/")


if __name__ == "__main__":
    train_logreg_baseline()
