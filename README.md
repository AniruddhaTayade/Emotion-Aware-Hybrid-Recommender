# 🧠 Emotion-Aware Hybrid Recommender System

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-ML-red?style=flat&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-DistilBERT-yellow?style=flat&logo=huggingface)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF%20%7C%20SVD-orange?style=flat&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

> A graduate-level ML system that classifies user emotions from free-text input and delivers
> personalized mental health resource recommendations using a hybrid NLP + collaborative filtering architecture.

---

## 🏆 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression (Baseline) | 0.2739 | 0.3733 | 0.2739 | 0.2699 |
| **DistilBERT (Ours)** | **0.4870** | **0.4736** | **0.4870** | **0.4652** |

> DistilBERT outperformed the baseline by **77% relative improvement** across 27 emotion categories on the GoEmotions corpus.

---

## 🏗️ System Architecture

```
User Free-Text Input
        ↓
DistilBERT Emotion Classifier
(fine-tuned on GoEmotions — 27 emotion categories)
        ↓
Hybrid Recommender System
  ├── TF-IDF Content-Based Filtering
  │     └── Cosine similarity between user query & resource descriptions
  └── SVD Collaborative Filtering
        └── Latent factor extraction from user-resource rating matrix
        ↓
Hybrid Score Aggregation
(final_score = 0.7 × content_score + 0.3 × collab_score)
        ↓
Top-K Mental Health Resource Recommendations
```

---

## ✨ Key Features

- **Emotion Detection** — Fine-tuned `distilbert-base-uncased` on GoEmotions classifying 27 distinct emotion categories using BERT WordPiece tokenization and Softmax output
- **Content-Based Filtering** — TF-IDF vectorization across 115 curated mental health resources (115 × 413 token matrix) with cosine similarity scoring
- **Collaborative Filtering** — Truncated SVD matrix factorization on synthetic user-resource ratings for latent item quality estimation
- **Hybrid Aggregation** — Weighted combination balancing semantic relevance and item quality for stable Top-K results
- **Model Benchmarking** — Full evaluation suite comparing DistilBERT vs Logistic Regression with confusion matrix analysis

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Emotion Classification | DistilBERT (Hugging Face Transformers) |
| Content-Based Filtering | TF-IDF + Cosine Similarity (scikit-learn) |
| Collaborative Filtering | Truncated SVD (scikit-learn) |
| Deep Learning Framework | PyTorch |
| Primary Dataset | GoEmotions (Google Research) |
| Resource Dataset | Custom rsrc.csv (115 resources) |
| Language | Python 3.8+ |

---

## 📁 Project Structure

```
Emotion-Aware-Hybrid-Recommender/
│
├── main.py                        # Entry point — runs full pipeline
├── gpu_test.py                    # GPU availability check
│
├── src/
│   ├── train_emotion_model.py     # Fine-tune DistilBERT on GoEmotions
│   ├── emotion_predictor.py       # Predict emotion from user input
│   ├── evaluate_emotion_model.py  # Evaluate DistilBERT performance
│   ├── evaluate_trained_model.py  # Full model evaluation suite
│   ├── evaluation.py              # Metrics: Accuracy, Precision, Recall, F1
│   ├── logreg_baseline.py         # Logistic Regression baseline model
│   ├── clean_goemotions.py        # Preprocess GoEmotions dataset
│   ├── build_recommender.py       # Build TF-IDF recommender
│   ├── collaborative.py           # SVD collaborative filtering
│   └── recommender.py             # Hybrid score aggregation
│
├── data/
│   ├── rsrc.csv                   # 115 curated mental health resources ✅ included
│   ├── ratings.csv                # Synthetic user-resource ratings ✅ included
│   ├── articleyt.csv              # Supplementary article data ✅ included
│   ├── goemotions_1.csv           # ⬇️ Download separately (see below)
│   ├── goemotions_2.csv           # ⬇️ Download separately (see below)
│   └── goemotions_3.csv           # ⬇️ Download separately (see below)
│
├── models/                        # ⬇️ Generated after training (not included)
│   └── distilbert_emotion/
│       ├── model.safetensors
│       ├── config.json
│       └── tokenizer files
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📥 Dataset Setup

### GoEmotions Dataset (Required for Training)

The GoEmotions dataset is **not included** due to file size. Download all three files and place them inside the `data/` folder:

```bash
wget https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv -P data/
wget https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv -P data/
wget https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv -P data/
```

Or download manually from: https://github.com/google-research/google-research/tree/master/goemotions

Then preprocess:

```bash
python src/clean_goemotions.py
```

### Pretrained DistilBERT Model

The fine-tuned model weights are **not included** due to size (~1.2GB).

- **Option 1 — Train from scratch** (~2 hours with GPU, ~8 hours CPU):

```bash
python src/train_emotion_model.py
```

- **Option 2 — Download pretrained weights:** [[Google Drive Link](https://drive.google.com/drive/folders/1rM-VqIfNnqfWlwPLKyB8HOB9odUrn8gt?usp=sharing)]

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU recommended — check availability with `python gpu_test.py`

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/AniruddhaTayade/Emotion-Aware-Hybrid-Recommender.git
cd Emotion-Aware-Hybrid-Recommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download GoEmotions dataset (see Dataset Setup above)

# 4. Preprocess the dataset
python src/clean_goemotions.py

# 5. Train the emotion classifier
python src/train_emotion_model.py

# 6. Run the full pipeline
python main.py
```

---

## 💻 Example Output

```
● Enter a sentence about how you feel:
> I can't tell whether I made the right choice because everything feels unclear.

==============================
● Emotion Prediction Result
==============================
Predicted Emotion: confusion
Confidence: 0.6851

● Filtering resources for: confusion
● Using Hybrid Recommender
----------------------------------
✦ Title: Grounding Practice For Overwhelm
■ Description: Guided meditation that helps calm the mind when thoughts feel scattered
🔗 Link: https://www.youtube.com/watch?v=wE292vsJcBY
Type: video | Emotion Tag: confusion
----------------------------------
✦ Title: Understanding Negative Emotions
■ Description: Overview of negative emotions and how to respond skillfully
🔗 Link: https://www.verywellmind.com/how-should-i-deal-with-negative-emotions
Type: article | Emotion Tag: confusion
----------------------------------
✦ Title: Breathing To Clear Your Mind
■ Description: Short video teaching paced breathing to calm confusion and worry
🔗 Link: https://www.youtube.com/watch?v=V6ru4eTnWJI
Type: video | Emotion Tag: confusion
```

---

## 📊 Evaluation Details

### Emotion Classifier

Both models were evaluated on Accuracy, Precision, Recall, F1-Score, and Confusion Matrix analysis across all 27 GoEmotions categories.

Key findings:
- DistilBERT showed stronger diagonal concentration in the confusion matrix — better class separation
- Class imbalance in GoEmotions impacted minority emotion accuracy for both models
- Multi-label to single-label conversion reduced emotional nuance but was necessary for supervised classification
- Despite moderate accuracy, DistilBERT provided sufficient emotion signal for downstream recommendation filtering

### Hybrid Recommender

- TF-IDF effectively captured semantic alignment between user queries and resource descriptions
- SVD latent factors provided stable item quality estimates even with synthetic rating data
- Hybrid aggregation (0.7 content + 0.3 collab) consistently outperformed either component alone

---

## ⚠️ Limitations

- Synthetic rating data limits real-world ecological validity
- Resource catalog is small (115 items) — expanding planned
- Class imbalance in GoEmotions affects minority emotion recall
- No multi-turn conversational memory
- Single-label conversion reduces emotional nuance
- Resource descriptions are manually curated and may carry bias

---

## 🔮 Future Work

- Replace synthetic ratings with real user interaction data
- Expand resource catalog beyond 115 items
- Implement multi-label emotion modeling
- Add conversational memory for multi-turn sessions
- Build a web interface (Flask/React)
- Deploy as a REST API

---

## 👤 Contributor

**Aniruddha Tayade**  
[LinkedIn](https://www.linkedin.com/in/aniruddhatayade/) | [GitHub](https://github.com/AniruddhaTayade)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Google Research — GoEmotions Dataset](https://github.com/google-research/google-research/tree/master/goemotions)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [scikit-learn](https://scikit-learn.org/)
- Florida International University — Graduate ML Course Project
