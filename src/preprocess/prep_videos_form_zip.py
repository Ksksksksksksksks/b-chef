import zipfile
import cv2
import pandas as pd
import os
from io import BytesIO
import tempfile
import subprocess

pd.set_option("display.max_rows", None)

base_dir = os.path.dirname(__file__)  # src/preprocess
zip_path = os.path.abspath(os.path.join(base_dir, "../../data/raw/video/mpii_video.zip"))
csv_path = os.path.abspath(os.path.join(base_dir, "../../data/raw/video/mpii_ground_truth.csv"))
clips_dir = os.path.abspath(os.path.join(base_dir, "../../data/processed/video_clips"))

os.makedirs(clips_dir, exist_ok=True)

df = pd.read_csv(csv_path, header=None,
                 names=["subject", "file_name", "start_frame", "end_frame", "category_id", "action"])


# print("All categories in ground truth:")
# print(df['action'].value_counts())

# only several, for mvp
wanted_categories = [
    "put in bowl",
    "cut slices",
    "cut dice",
    "cut apart",
    "peel",
    "stir",
    "put on cutting-board",
    "pour",
    "whisk",
    "grate",
    "cut stripes",
    "open egg",
    "mix",
    "put in pan/pot"
]


df = df[df['action'].isin(wanted_categories)]

print(f"Will get {len(df)} clips")

with zipfile.ZipFile(zip_path, 'r') as zf:
    for _, row in df.iterrows():
        csv_file_name = row['file_name']
        category = row['action'].replace(' ', '_')
        video_name_in_zip = None
        for name in zf.namelist():
            if name.endswith('.avi') and csv_file_name in name:
                video_name_in_zip = name
                break

        if not video_name_in_zip:
            print(f"Not found video {csv_file_name} in ZIP")
            continue
        print(f"Processing video {csv_file_name}, category {category}")

        category_dir = os.path.join(clips_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        with zf.open(video_name_in_zip) as video_file:
            tmp_path = os.path.join(tempfile.gettempdir(), "temp.avi")
            with open(tmp_path, "wb") as f:
                f.write(video_file.read())

            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            start_frame = int(row['start_frame'])
            end_frame = int(row['end_frame'])

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

            out_name = f"{csv_file_name}_{start_frame}_{end_frame}.avi"
            out_path = os.path.join(category_dir, out_name)

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

        try:
            subprocess.run(["dvc", "add", out_path], check=True)
            os.remove(out_path)
            print(f"Added to DVC and removed local file: {out_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error adding {out_name} to DVC: {e}")