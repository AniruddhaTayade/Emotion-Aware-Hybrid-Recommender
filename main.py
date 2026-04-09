import sys
import os

# Add src folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from recommender import ResourceRecommender
from emotion_predictor import EmotionPredictor
from collaborative import CollaborativeRecommender   # NEW


def main():
    print("\n==============================")
    print("🧠 Emotion-Based Recommendation System")
    print("==============================")

    # Load models
    print("\n📌 Loading models...")
    rec = ResourceRecommender("data/rsrc.csv")                 # TF-IDF recommender
    clf = EmotionPredictor("models/distilbert_emotion")        # DistilBERT model
    collab = CollaborativeRecommender("data/ratings.csv")      # NEW SVD recommender

    # Take user input
    print("\n💬 Enter a sentence about how you feel:")
    user_text = input("> ")

    if not user_text.strip():
        print("⚠ Please enter some text.")
        sys.exit()

    # Emotion Prediction
    predicted_emotion, confidence = clf.predict(user_text)

    print("\n==============================")
    print("🧠 Emotion Prediction Result")
    print("==============================")
    print(f"Predicted Emotion: {predicted_emotion}")
    print(f"Confidence: {confidence:.4f}")

    # Hybrid Recommendations
    print("\n==============================")
    print("📚 Hybrid Recommended Resources")
    print("==============================")

    results = rec.recommend(
        user_text, 
        predicted_emotion, 
        collab=collab,        # NEW hybrid scoring
        top_k=5
    )

    if results.empty:
        print("⚠ No recommendations found.")
    else:
        for idx, row in results.iterrows():
            print("\n----------------------------------")
            print(f"🔹 Title: {row['title']}")
            print(f"📄 Description: {row['description']}")
            print(f"🔗 Link: {row['link']}")
            print(f"🏷 Type: {row['type']}")
            print(f"❤️ Emotion Tag: {row['emotion']}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
