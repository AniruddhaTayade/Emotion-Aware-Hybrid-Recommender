import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

class EmotionPredictor:
    def __init__(self, model_path="models/distilbert_emotion"):
        print("📌 Loading emotion model...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        predicted_id = torch.argmax(probs, dim=1).item()

        # id2label is already inside the model config
        predicted_label = self.model.config.id2label[predicted_id]

        return predicted_label, probs[0][predicted_id].item()
