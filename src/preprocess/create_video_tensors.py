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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        dvc_path = os.path.join(self.root_dir, row["label"], row["filename"] + ".dvc")
        label = row["label"]

        tmp_path = f"temp_{os.path.basename(dvc_path).replace('.dvc', '.avi')}"

        try:
            try:
                with dvc.api.open(dvc_path, mode="rb") as f:
                    with open(tmp_path, "wb") as temp_video:
                        temp_video.write(f.read())
            except Exception as e_remote:
                import yaml
                with open(dvc_path, "r") as f:
                    dvc_meta = yaml.safe_load(f)
                md5 = dvc_meta["outs"][0]["md5"]
                repo_root = os.path.abspath(os.path.join(__file__, "../../.."))
                cache_dir = os.path.join(repo_root, ".dvc", "cache", "files", "md5")
                cache_path = os.path.join(cache_dir, md5[:2], md5[2:])


                if not os.path.exists(cache_path):
                    print(f"❌ Video are not in remote: {dvc_path}")
                    print(f"❌  nor in cache: {cache_path}")
                    raise FileNotFoundError
                with open(cache_path, "rb") as f_cache, open(tmp_path, "wb") as f_tmp:
                    f_tmp.write(f_cache.read())

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
                print(f"⚠️ Empty video: {dvc_path}")
                raise ValueError("No frames read")

            video_tensor = torch.stack(frames)
            return video_tensor, label


        except Exception as e:
            print(f"❌ Error processing {dvc_path}: {e}")
            dummy = torch.zeros((self.max_frames, 3, 224, 224))
            return dummy, "error"


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)  # src/preprocess

    CLIPS_DIR = os.path.abspath(os.path.join(base_dir, "../../data/processed/video_clips"))
    OUTPUT_DIR = os.path.abspath(os.path.join(base_dir, "../../data/processed/video_dataset"))
    SAVED_DIR = os.path.join(OUTPUT_DIR, "tensors")
    os.makedirs(SAVED_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])

    datasets = {
        "train": DVCVideoDataset(os.path.join(OUTPUT_DIR, "train.csv"), CLIPS_DIR, transform),
        "val":   DVCVideoDataset(os.path.join(OUTPUT_DIR, "val.csv"),   CLIPS_DIR, transform),
        "test":  DVCVideoDataset(os.path.join(OUTPUT_DIR, "test.csv"),  CLIPS_DIR, transform),
    }

    for split, dataset in datasets.items():
        print(f"💾 Saving tensors for {split} split...")
        split_dir = os.path.join(SAVED_DIR, split)
        os.makedirs(split_dir, exist_ok=True)

        for i in tqdm(range(len(dataset))):
            video_tensor, label = dataset[i]
            filename = f"{i}_{label}.pt"
            torch.save((video_tensor, label), os.path.join(split_dir, filename))

    print("✅ All tensors saved successfully!")

