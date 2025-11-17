import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import *
import cv2
from pytorchvideo.models.hub import slowfast_r50
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("src/video_model/all_classes.txt", "r") as f:
    all_classes = [line.strip() for line in f.readlines()]


# =========================
# 1. MODEL LOADING
# =========================
class SlowFastWithFeatures(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        in_features = base_model.blocks[-1].proj.in_features
        base_model.blocks[-1].proj = nn.Linear(in_features, num_classes)

    def forward(self, inputs):
        # inputs = [slow, fast]
        return self.base_model(inputs)


def load_video_model(lora_path: str, num_classes: int = 25):
    print("[INFO] Loading SlowFast backbone...")

    base = slowfast_r50(pretrained=False)

    print("[INFO] Base model loaded.")

    model = SlowFastWithFeatures(base, num_classes)

    print(f"[INFO] Loading LoRA weights from {lora_path}...")
    state = torch.load(lora_path, map_location=DEVICE)
    model.load_state_dict(state, strict=False)

    print("[INFO] Model ready.")
    model.eval()
    return model.to(DEVICE).eval()


# =========================
# 2. VIDEO PREPROCESSING
# =========================
def preprocess_video(path, num_frames=32):
    print("[INFO] Preprocessing video...")

    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Total frames: {total}")

    idxs = torch.linspace(0, total - 1, num_frames).long().tolist()

    frames = []
    pos = 0
    target = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if pos == idxs[target]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            target += 1
            if target >= len(idxs):
                break
        pos += 1

    cap.release()

    print(f"[INFO] Collected frames: {len(frames)}")

    tf = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize([0.45]*3, [0.225]*3)
    ])

    frames = [Image.fromarray(f) for f in frames]

    return torch.stack([tf(f) for f in frames])


# =========================
# 3. SLOWFAST INPUT PACK
# =========================
def pack_slowfast(frames, alpha=4):
    print("[INFO] Packing SlowFast inputs...")
    fast = frames
    slow = frames[::alpha]

    slow = slow.permute(1,0,2,3).unsqueeze(0)  # [B,C,T,H,W]
    fast = fast.permute(1,0,2,3).unsqueeze(0)

    return [slow.to(DEVICE), fast.to(DEVICE)]


# =========================
# 4. MAIN INFERENCE
# =========================
def run_inference(model, video_path, topk=5):
    frames = preprocess_video(video_path)
    inputs = pack_slowfast(frames)

    print("[INFO] Running model inference...")

    with torch.no_grad():
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)[0]

    print("[INFO] Inference complete.")

    vals, idxs = probs.topk(topk)

    return {
        "logits": logits[0].cpu(),
        "probs": probs.cpu(),
        "topk_indices": idxs.cpu(),
        "topk_probs": vals.cpu(),
    }


# =========================
# 5. CLI MODE
# =========================
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--weights", default="src/video_model/best_lora_model.pth")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    model = load_video_model(args.weights)
    out = run_inference(model, args.video, args.topk)

    print("Top-k predictions:")
    for idx, pr in zip(out["topk_indices"], out["topk_probs"]):
        class_name = all_classes[int(idx)]
        print(f"{int(idx)}\t{class_name}\t{float(pr):.4f}")


