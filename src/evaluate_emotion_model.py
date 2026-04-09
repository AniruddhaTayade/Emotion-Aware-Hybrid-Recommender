import pandas as pd
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def evaluate_model():
    print("📌 Loading cleaned dataset...")
    df = pd.read_csv("data/cleaned_goemotions.csv")

    labels = sorted(df["emotion"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    df["labels"] = df["emotion"].map(label2id)

    dataset = Dataset.from_pandas(df[["text", "labels"]])
    dataset = dataset.train_test_split(test_size=0.1)

    print("📌 Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("models/distilbert_emotion")

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    print("📌 Loading trained model...")
    model = DistilBertForSequenceClassification.from_pretrained("models/distilbert_emotion")

    # Dummy training args
    training_args = TrainingArguments(
        output_dir="models/tmp_eval",
        per_device_eval_batch_size=16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics
    )

    print("🔍 Running evaluation...")
    results = trainer.evaluate()

    print("\n📊 Evaluation Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    evaluate_model()
