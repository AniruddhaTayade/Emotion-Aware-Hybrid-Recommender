from recommender import ResourceRecommender
from emotion_predictor import EmotionPredictor

# Load models
rec = ResourceRecommender("data/rsrc.csv")
clf = EmotionPredictor("models/distilbert_emotion")

# User input
sample_text = "I feel sick just thinking about what they did."

# Predict emotion
predicted_emotion, confidence = clf.predict(sample_text)

print("\n🧠 Predicted Emotion:", predicted_emotion)
print("Confidence:", round(confidence, 4))

# Get recommendations
print("\n📚 Recommendations:")
results = rec.recommend(sample_text, predicted_emotion)

print(results)
