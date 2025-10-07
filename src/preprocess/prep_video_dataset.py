import os
import pandas as pd
from sklearn.model_selection import train_test_split

# PATHS
base_dir = os.path.dirname(__file__)  # src/preprocess

CLIPS_DIR = os.path.abspath(os.path.join(base_dir, "../../data/processed/video_clips"))
OUTPUT_DIR = os.path.abspath(os.path.join(base_dir, "../../data/processed/video_dataset"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

rows = []


for label in sorted(os.listdir(CLIPS_DIR)):
    cat_dir = os.path.join(CLIPS_DIR, label)
    if not os.path.isdir(cat_dir):
        continue

    videos = [f for f in os.listdir(cat_dir) if f.endswith(".avi.dvc")]
    for file in videos:
        filename = file.replace(".dvc", "")
        rows.append({"filename": filename, "label": label})


df = pd.DataFrame(rows)
print(f"📦 Overall clips: {len(df)} in {df['label'].nunique()} categories")

summary = df.groupby("label").size().reset_index(name="count")
summary.to_csv(os.path.join(OUTPUT_DIR, "labels_summary.csv"), index=False)
print("✅ labels_summary.csv done")

train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

print(f"✅ Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"📂 CSV saved into {OUTPUT_DIR}")
