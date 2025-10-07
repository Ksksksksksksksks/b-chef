import os
import cv2
import torch
import dvc.api
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


class DVCVideoDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, max_frames=32):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.max_frames = max_frames

        print(f"📂 Initialized datset: {os.path.basename(csv_file)} | #clips: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_path = os.path.join(self.root_dir, row["label"], row["filename"] + ".dvc")
        label = row["label"]

        try:
            with dvc.api.open(video_path, mode="rb") as f:
                tmp_path = f"temp_{os.path.basename(video_path)}"
                with open(tmp_path, "wb") as temp_video:
                    temp_video.write(f.read())

            cap = cv2.VideoCapture(tmp_path)
            frames = []
            while len(frames) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)

            cap.release()
            os.remove(tmp_path)

            if not frames:
                print(f"⚠️ Empty video: {video_path}")
                raise ValueError("No frames read")

            video_tensor = torch.stack(frames)
            return video_tensor, label

        except Exception as e:
            print(f"❌ Error during downloading {video_path}: {e}")
            dummy = torch.zeros((self.max_frames, 3, 224, 224))
            return dummy, "error"


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)  # src/preprocess

    CLIPS_DIR = os.path.abspath(os.path.join(base_dir, "../../data/processed/video_clips"))
    OUTPUT_DIR = os.path.abspath(os.path.join(base_dir, "../../data/processed/video_dataset"))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])

    train_dataset = DVCVideoDataset(os.path.join(OUTPUT_DIR, "train.csv"), CLIPS_DIR, transform)
    val_dataset   = DVCVideoDataset(os.path.join(OUTPUT_DIR, "val.csv"),   CLIPS_DIR, transform)
    test_dataset  = DVCVideoDataset(os.path.join(OUTPUT_DIR, "test.csv"),  CLIPS_DIR, transform)

    print(f"\n✅ Datasets are ok:")
    print(f"   train: {len(train_dataset)} | val: {len(val_dataset)} | test: {len(test_dataset)}\n")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False)

    print("🚀 Check 1st videos from  train...")
    for i, (frames, label) in enumerate(tqdm(train_loader)):
        print(f"[train] #{i} | shape={frames.shape} | label={label}")
        if i == 3:
            break

    print("✅ all good!")
