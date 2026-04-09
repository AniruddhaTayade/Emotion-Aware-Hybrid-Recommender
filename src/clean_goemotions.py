import pandas as pd
import glob

def clean_goemotions():
    print("\n🔄 Loading GoEmotions files...")

    # Load all goemotions_*.csv files automatically
    files = glob.glob("data/goemotions_*.csv")
    if not files:
        print("❌ No goemotions_*.csv files found in data/")
        return
    
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"📌 Loaded {len(dfs)} files with total {len(df)} rows.")

    # Columns that are NOT emotion labels
    ignore_cols = [
        "text", "id", "author", "subreddit", "link_id", "parent_id",
        "created_utc", "rater_id", "example_very_unclear"
    ]

    # All other columns = emotion labels
    emotion_cols = [col for col in df.columns if col not in ignore_cols]

    cleaned_rows = []
    dropped = 0

    print("🔍 Cleaning rows...")

    for _, row in df.iterrows():
        # Find all emotion labels with value = 1
        labels = [emotion for emotion in emotion_cols if row[emotion] == 1]

        if labels:
            cleaned_rows.append([row["text"], labels[0]])  # pick first emotion
        else:
            dropped += 1

    print(f"🧹 Dropped {dropped} rows with no emotion labels.")

    # Save cleaned dataframe
    cleaned_df = pd.DataFrame(cleaned_rows, columns=["text", "emotion"])
    cleaned_df.to_csv("data/cleaned_goemotions.csv", index=False)

    print(f"\n✅ Saved cleaned dataset → data/cleaned_goemotions.csv")
    print(f"📊 Total cleaned rows: {len(cleaned_df)}\n")

if __name__ == "__main__":
    clean_goemotions()
