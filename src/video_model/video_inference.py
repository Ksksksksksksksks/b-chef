import torch
import torch.nn as nn
import cv2
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from pytorchvideo.models.hub import slowfast_r50
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 1. MODEL + LoRA
# =========================
class LinearLoRA(nn.Module):
    def __init__(self, linear_layer, r=4, dropout=0.1):
        super().__init__()
        self.linear = linear_layer
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        self.lora = nn.Sequential(
            nn.Linear(in_features, r, bias=False),
            nn.Linear(r, out_features, bias=False),
            nn.Dropout(dropout)
        )
        for p in self.lora.parameters():
            p.requires_grad = True
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.linear(x) + self.lora(x)
        print(f"[LOG] LinearLoRA forward: input {x.shape}, output {out.shape}")
        return out

class Conv3dLoRA(nn.Module):
    def __init__(self, conv_layer, r=4, dropout=0.1):
        super().__init__()
        self.conv = conv_layer
        self.lora = nn.Sequential(
            nn.Conv3d(conv_layer.in_channels, r, kernel_size=1, bias=False),
            nn.Conv3d(r, conv_layer.out_channels, kernel_size=1, bias=False),
            nn.Dropout3d(dropout)
        )
        for p in self.lora.parameters():
            p.requires_grad = True
        for p in self.conv.parameters():
            p.requires_grad = False

    def forward(self, x):
        conv_out = self.conv(x)
        lora_out = self.lora(x)
        if lora_out.shape[2:] != conv_out.shape[2:]:
            lora_out = torch.nn.functional.interpolate(lora_out, size=conv_out.shape[2:], mode='trilinear', align_corners=False)
        out = conv_out + lora_out
        print(f"[LOG] Conv3dLoRA forward: input {x.shape}, output {out.shape}")
        return out

class SlowFastWithLoRA(nn.Module):
    def __init__(self, base_model, r=4, alpha=16, dropout=0.1, target_modules=None):
        super().__init__()
        self.base_model = base_model
        if target_modules is None:
            target_modules = ["proj"]
        for name, module in self.base_model.named_modules():
            for target_name in target_modules:
                if target_name in name:
                    if isinstance(module, nn.Linear):
                        self._replace_module(name, LinearLoRA(module, r=r, dropout=dropout))
                    elif isinstance(module, nn.Conv3d):
                        self._replace_module(name, Conv3dLoRA(module, r=r, dropout=dropout))
        self.to(next(base_model.parameters()).device)

    def _replace_module(self, module_name, new_module):
        parts = module_name.split('.')
        mod = self.base_model
        for p in parts[:-1]:
            mod = getattr(mod, p)
        setattr(mod, parts[-1], new_module)
        print(f"[LOG] Replaced module: {module_name} -> {new_module.__class__.__name__}")

    def forward(self, inputs):
        print(f"[LOG] Forward called with inputs: {[x.shape for x in inputs]}")
        out = self.base_model(inputs)
        print(f"[LOG] Forward output: {out.shape}")
        return out

def load_lora_model(weights_path: str, num_classes: int = 25):
    print(f"[LOG] Loading base SlowFast model with {num_classes} classes")
    base = slowfast_r50(pretrained=False)
    in_features = base.blocks[-1].proj.in_features
    base.blocks[-1].proj = nn.Linear(in_features, num_classes)
    model = SlowFastWithLoRA(base, target_modules=['conv', 'conv_fast_to_slow', 'proj'])
    print(f"[LOG] Loading LoRA weights from {weights_path}")
    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)
    print(f"[LOG] Model loaded and set to eval mode")
    return model.eval().to(DEVICE)

# =========================
# 2. VIDEO PREPROCESSING
# =========================
def preprocess_video(path, num_frames=32):
    print(f"[LOG] Opening video {path}")
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[LOG] Total frames in video: {total}")
    idxs = torch.linspace(0, total - 1, num_frames).long().tolist()
    frames = []
    pos = 0
    target = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if pos == idxs[target]:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            target += 1
            if target >= len(idxs): break
        pos += 1
    cap.release()
    print(f"[LOG] Extracted {len(frames)} frames")

    tf = Compose([Resize(256), CenterCrop(224), ToTensor(), Normalize([0.45]*3, [0.225]*3)])
    frames = [Image.fromarray(f) for f in frames]
    frames_tensor = torch.stack([tf(f) for f in frames])
    print(f"[LOG] Frames tensor shape: {frames_tensor.shape}")
    return frames_tensor

def pack_slowfast(frames, alpha=4):
    fast = frames
    slow = frames[::alpha]
    slow = slow.permute(1,0,2,3).unsqueeze(0)
    fast = fast.permute(1,0,2,3).unsqueeze(0)
    print(f"[LOG] Packed slow shape: {slow.shape}, fast shape: {fast.shape}")
    return [slow.to(DEVICE), fast.to(DEVICE)]

# # =========================
# # 3. INFERENCE
# # =========================
# def run_inference(model, video_path, topk=5):
#     frames = preprocess_video(video_path)
#     inputs = pack_slowfast(frames)
#     with torch.no_grad():
#         logits = model(inputs)
#         probs = torch.softmax(logits, dim=1)[0]
#     vals, idxs = probs.topk(topk)
#     print(f"[LOG] Top-{topk} probabilities: {vals}")
#     return {"logits": logits[0].cpu(), "probs": probs.cpu(), "topk_indices": idxs.cpu(), "topk_probs": vals.cpu()}

# =========================
# 3. INFERENCE (WINDOW-BASED)
# =========================

def sliding_windows(frames, window_size=32, stride=16):
    windows = []
    L = frames.shape[0]

    for i in range(0, L - window_size + 1, stride):
        windows.append(frames[i:i+window_size])

    if not windows:
        windows = [frames]

    print(f"[LOG] Total windows: {len(windows)}")
    return windows


def run_inference(model, video_path, topk=5):
    frames = preprocess_video(video_path)

    windows = sliding_windows(frames, window_size=32, stride=16)
    all_logits = []

    with torch.no_grad():
        for i, win in enumerate(windows):
            inputs = pack_slowfast(win)
            logits = model(inputs)              # shape: [1, num_classes]
            all_logits.append(logits)

            probs = F.softmax(logits, dim=1)[0]
            vals, idxs = probs.topk(topk)
            print(f"[LOG] Window {i}: top-{topk} probs = {vals.cpu().numpy()}, idx = {idxs.cpu().numpy()}")

    final_logits = torch.stack(all_logits).mean(dim=0)   # [1, num_classes]
    final_probs = F.softmax(final_logits, dim=1)[0]

    vals, idxs = final_probs.topk(topk)

    print(f"\n[LOG] Final averaged probabilities:")
    print(final_probs.cpu().numpy())

    return {
        "logits": final_logits[0].cpu(),
        "probs": final_probs.cpu(),
        "topk_indices": idxs.cpu(),
        "topk_probs": vals.cpu()
    }


# =========================
# 4. CLI
# =========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--weights", default="src/video_model/best_lora_model.pth")
    parser.add_argument("--classes", default="src/video_model/all_classes.txt")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    with open(args.classes) as f:
        all_classes = [line.strip() for line in f.readlines()]

    model = load_lora_model(args.weights, num_classes=len(all_classes))
    out = run_inference(model, args.video, args.topk)

    print("\nTop-k predictions:")
    for idx, pr in zip(out["topk_indices"], out["topk_probs"]):
        print(f"{int(idx)}\t{all_classes[int(idx)]}\t{float(pr):.10f}")
