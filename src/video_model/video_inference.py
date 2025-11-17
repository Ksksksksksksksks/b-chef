import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import *
import cv2
from pytorchvideo.models.hub import slowfast_r50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 1. MODEL LOADING
# =========================
class SlowFastWithFeatures(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = base_model.fc
        self.num_classes = num_classes

    def forward(self, x):
        feats = self.backbone(x)         # (N, C, T, H, W)
        pooled = self.pool(feats)        # (N, C, 1,1,1)
        pooled = pooled.flatten(1)       # (N, C)
        logits = self.fc(pooled)         # (N, num_classes)
        return feats, pooled, logits


def load_video_model(lora_path: str, num_classes: int = 25):

    base = slowfast_r50(pretrained=True)

    base.fc = nn.Linear(base.fc.in_features, num_classes)

    model = SlowFastWithFeatures(base, num_classes)

    state = torch.load(lora_path, map_location=DEVICE)
    model.load_state_dict(state, strict=False)

    return model.to(DEVICE).eval()


# =========================
# 2. VIDEO PREPROCESSING
# =========================
def preprocess_video(path, num_frames=32):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

    tf = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize([0.45]*3, [0.225]*3)
    ])

    return torch.stack([tf(f) for f in frames])


# =========================
# 3. SLOWFAST INPUT PACK
# =========================
def pack_slowfast(frames, alpha=4):
    fast = frames             # (T, C, H, W)
    slow = frames[::alpha]

    slow = slow.permute(1,0,2,3).unsqueeze(0)
    fast = fast.permute(1,0,2,3).unsqueeze(0)

    return [slow.to(DEVICE), fast.to(DEVICE)]


# =========================
# 4. MAIN INFERENCE
# =========================
def run_inference(model, video_path, topk=5):
    frames = preprocess_video(video_path)
    inputs = pack_slowfast(frames)

    with torch.no_grad():
        feats, pooled, logits = model(inputs)
        probs = torch.softmax(logits, dim=1)[0]

    vals, idxs = probs.topk(topk)

    return {
        "features_raw": feats[0].cpu(),       # (C, T', H', W')
        "features_vector": pooled[0].cpu(),   # (C,)
        "logits": logits[0].cpu(),            # (num_classes,)
        "probs": probs.cpu(),                 # (num_classes,)
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
    parser.add_argument("--weights", default="best_lora_model.pth")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    model = load_video_model(args.weights)
    out = run_inference(model, args.video, args.topk)

    print("Top-k predictions:")
    for idx, pr in zip(out["topk_indices"], out["topk_probs"]):
        print(f"{int(idx)}\t{float(pr):.4f}")
