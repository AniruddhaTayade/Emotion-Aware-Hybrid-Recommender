import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_trained_model():

    print("📌 Loading cleaned dataset...")
    df = pd.read_csv("data/cleaned_goemotions.csv")

    # Rebuild label mapping (must be identical to training!)
    labels = sorted(df["emotion"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    df["labels"] = df["emotion"].map(label2id)

    # Create HF dataset
    dataset = Dataset.from_pandas(df[["text", "labels"]])
    dataset = dataset.train_test_split(test_size=0.1)

    print("📌 Loading tokenizer & trained model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("models/distilbert_emotion")
    model = DistilBertForSequenceClassification.from_pretrained("models/distilbert_emotion")
    model.eval()

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    print("🔍 Running evaluation...")
    test_loader = torch.utils.data.DataLoader(dataset["test"], batch_size=16)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )

    cm = confusion_matrix(all_labels, all_preds)

    print("\n==============================")
    print("📊 MODEL PERFORMANCE RESULTS")
    print("==============================")
    print(f"✔ Accuracy:  {accuracy:.4f}")
    print(f"✔ Precision: {precision:.4f}")
    print(f"✔ Recall:    {recall:.4f}")
    print(f"✔ F1 Score:  {f1:.4f}")

    print("\n🧩 Confusion Matrix:")
    print(cm)

    print("\n📝 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=labels))

    # ---------------------------------------------------------
    # ⭐ SAVE CONFUSION MATRIX AS IMAGE WITH EMOTION LABELS
    # ---------------------------------------------------------
    plt.figure(figsize=(22, 18))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title("Confusion Matrix (Emotion Labels)", fontsize=18)
    plt.xlabel("Predicted Label", fontsize=16)
    plt.ylabel("True Label", fontsize=16)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = "models/confusion_matrix.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"\n🖼 Confusion matrix saved → {save_path}")



if __name__ == "__main__":
    evaluate_trained_model()
