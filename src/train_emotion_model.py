import os
import pandas as pd
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import torch


def train_distilbert():

    print("📌 Loading dataset...")
    df = pd.read_csv("data/cleaned_goemotions.csv")

    # Label Encoding
    labels = sorted(df["emotion"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    df["labels"] = df["emotion"].map(label2id)

    # Create HF dataset
    dataset = Dataset.from_pandas(df[["text", "labels"]])
    dataset = dataset.train_test_split(test_size=0.1)

    print("🔄 Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    print("🤖 Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    # Auto GPU selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Using device: {device}")
    model.to(device)

    # Training arguments (compatible with Transformers 4.57+)
    training_args = TrainingArguments(
        output_dir="models/distilbert_emotion",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=200,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",     # <-- correct for v4.57+
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        report_to="none",          # disable wandb
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    print("🚀 Starting training...")
    trainer.train()

    # Save output
    print("💾 Saving model...")
    trainer.save_model("models/distilbert_emotion")
    tokenizer.save_pretrained("models/distilbert_emotion")

    print("✅ Training complete! Model saved to: models/distilbert_emotion/")


if __name__ == "__main__":
    train_distilbert()
