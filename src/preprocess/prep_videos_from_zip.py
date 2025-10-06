import zipfile
import cv2
import pandas as pd
import os
import tempfile
import subprocess

pd.set_option("display.max_rows", None)

# === Paths ===
base_dir = os.path.dirname(__file__)  # src/preprocess
zip_path = os.path.abspath(os.path.join(base_dir, "../../data/raw/video/mpii_video.zip"))
csv_path = os.path.abspath(os.path.join(base_dir, "../../data/raw/video/mpii_ground_truth.csv"))
actions_path = os.path.abspath(os.path.join(base_dir, "../../data/raw/video/mpii_actions_suitable.txt"))
clips_dir = os.path.abspath(os.path.join(base_dir, "../../data/processed/video_clips"))
os.makedirs(clips_dir, exist_ok=True)

# === Config ===
MAX_CLIP_SECONDS = 15
FRAME_EXTENSION = ".avi"
DVC_EXTENSION = ".avi.dvc"


# === Load ground truth ===
df = pd.read_csv(csv_path, header=None,
                 names=["subject", "file_name", "start_frame", "end_frame", "category_id", "action"])

# Load needed categories from text file
with open(actions_path, "r", encoding="utf-8") as f:
    needed_categories = [line.strip() for line in f if line.strip()]

df = df[df['action'].isin(needed_categories)]

print(f"\nFiltered to {len(df)} clips to extract.\n")

# === Extract clips from ZIP ===
with zipfile.ZipFile(zip_path, 'r') as zf:
    for _, row in df.iterrows():
        csv_file_name = row['file_name']
        category = row['action'].replace(' ', '_')
        category_dir = os.path.join(clips_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        out_name = f"{csv_file_name}_{row['start_frame']}_{row['end_frame']}.avi"
        out_path = os.path.join(category_dir, out_name)
        dvc_path = out_path + ".dvc"

        if os.path.exists(out_path) or os.path.exists(dvc_path):
            print(f"Skipping {out_name} (already exists)")
            continue

        video_name_in_zip = next((n for n in zf.namelist() if n.endswith('.avi') and csv_file_name in n), None)
        if not video_name_in_zip:
            print(f"Not found video {csv_file_name} in ZIP")
            continue
        print(f"Processing {out_name} (category={category})")

        with zf.open(video_name_in_zip) as video_file:
            tmp_path = os.path.join(tempfile.gettempdir(), "temp.avi")
            with open(tmp_path, "wb") as f:
                f.write(video_file.read())

        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(row['start_frame'])
        end_frame = int(row['end_frame'])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = (end_frame - start_frame) / fps

        # === handle too long clips ===
        if duration_sec > MAX_CLIP_SECONDS:
            end_frame = start_frame + int(MAX_CLIP_SECONDS * fps)
            print(f"  Clip too long ({duration_sec:.1f}s). Trimmed to {MAX_CLIP_SECONDS}s.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        current_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if start_frame <= current_frame <= end_frame:
                out.write(frame)
            current_frame += 1
            if current_frame > end_frame:
                break

        cap.release()
        out.release()
        os.remove(tmp_path)

        # === add to DVC ===
        try:
            subprocess.run(["dvc", "add", out_path], check=True)
            os.remove(out_path)
            print(f"  Added to DVC and removed local file: {out_name}\n")
        except subprocess.CalledProcessError as e:
            print(f"  Error adding {out_name} to DVC: {e}\n")
