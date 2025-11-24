# unified_inference.py
import os
import tempfile
import shutil
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import logging
from rich.logging import RichHandler
from rich import print
import cv2
from PIL import Image
import numpy as np
import torch

# Configure rich logging
logging.basicConfig(level="INFO", format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("unified_inference")

# Default model paths
VIDEO_WEIGHTS = str(Path(__file__).resolve().parents[1] / "models_weights" / "video_model" / "best_lora_model.pth")
VIDEO_CLASSES = str(Path(__file__).resolve().parents[1] / "models_weights" / "video_model" / "all_classes.txt")
PHOTO_MODEL_DIR = str(Path(__file__).resolve().parents[1] / "models_weights" / "food_finetune" / "best_model")
PHOTO_MAPPINGS = str(Path(__file__).resolve().parents[1] / "models_weights" / "food_finetune" / "mappings.json")

# Lazy imports for heavy libs
_video_model = None
_video_classes = None
_photo_model = None
_photo_processor = None

def is_image(path: str) -> bool:
    return path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))

def is_video(path: str) -> bool:
    return path.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))

# -------------------------
# Helpers: frames extraction
# -------------------------
def extract_frames_to_files(video_path: str, num: int = 5) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total == 0:
        cap.release()
        raise RuntimeError("Cannot read video or no frames")

    idxs = np.linspace(0, total - 1, num).astype(int).tolist()
    tmpdir = tempfile.mkdtemp(prefix="bchef_frames_")
    saved = []
    pos = 0
    target = 0
    while cap.isOpened() and target < len(idxs):
        ret, frame = cap.read()
        if not ret:
            break
        if pos == idxs[target]:
            fname = os.path.join(tmpdir, f"frame_{target:03d}.jpg")
            cv2.imwrite(fname, frame)
            saved.append(fname)
            target += 1
        pos += 1
    cap.release()
    if not saved:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            fname = os.path.join(tmpdir, "frame_000.jpg")
            cv2.imwrite(fname, frame)
            saved.append(fname)
    logger.info(f"Extracted {len(saved)} frames to {tmpdir}")
    return saved

# -------------------------
# Video model wrapper
# -------------------------
def load_video_model(weights_path: str = VIDEO_WEIGHTS, classes_path: str = VIDEO_CLASSES):
    global _video_model, _video_classes
    if _video_model is not None:
        return _video_model, _video_classes

    import importlib.util, sys
    mod_path = Path(__file__).resolve().parents[0] / "video_inference.py"
    spec = importlib.util.spec_from_file_location("video_inference", str(mod_path))
    vi = importlib.util.module_from_spec(spec)
    sys.modules["video_inference"] = vi
    spec.loader.exec_module(vi)

    with open(classes_path, "r", encoding="utf-8") as f:
        classes = [l.strip() for l in f if l.strip()]

    model = vi.load_lora_model(weights_path, num_classes=len(classes))
    _video_model = model
    _video_classes = classes
    logger.info(f"Video model loaded: weights={weights_path}, classes={len(classes)}")
    return _video_model, _video_classes

def run_video_inference_wrapper(video_path: str, topk: int = 5, weights_path: str = VIDEO_WEIGHTS, classes_path: str = VIDEO_CLASSES):
    model, classes = load_video_model(weights_path, classes_path)
    import sys
    vi = sys.modules.get("video_inference")
    out = vi.run_inference(model, video_path, topk=topk)
    # unified format
    topk_indices = [int(x) for x in out["topk_indices"].cpu().numpy()]
    topk_probs = [float(x) for x in out["topk_probs"].cpu().numpy()]
    scores = {classes[i]: p for i, p in zip(topk_indices, topk_probs)}
    top1 = classes[topk_indices[0]]
    return {"top1": top1, "scores": scores, "raw": out}

# -------------------------
# Photo model wrapper
# -------------------------
def load_photo_model():
    global _photo_model, _photo_processor
    if _photo_model is not None:
        return _photo_model, _photo_processor

    import importlib.util, sys
    mod_path = Path(__file__).resolve().parents[0] / "photo_inference.py"
    spec = importlib.util.spec_from_file_location("photo_inference", str(mod_path))
    pi = importlib.util.module_from_spec(spec)
    sys.modules["photo_inference"] = pi
    spec.loader.exec_module(pi)

    _photo_model = pi.food_model
    _photo_processor = pi.food_processor
    return _photo_model, _photo_processor

def run_photo_inference_wrapper(image_path: str, topk: int = 5):
    import importlib.util, sys
    pi = sys.modules.get("photo_inference")
    if pi is None:
        mod_path = Path(__file__).resolve().parents[0] / "photo_inference.py"
        spec = importlib.util.spec_from_file_location("photo_inference", str(mod_path))
        pi = importlib.util.module_from_spec(spec)
        sys.modules["photo_inference"] = pi
        spec.loader.exec_module(pi)

    # extract logits for top-k
    image = Image.open(image_path)
    inputs = pi.food_processor(image, return_tensors="pt")
    with torch.no_grad():
        logits = pi.food_model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
        vals, idxs = probs.topk(topk)
        topk_labels = [pi.food_model.config.id2label[idx.item()] for idx in idxs]
        scores = {l: float(v) for l, v in zip(topk_labels, vals)}
    top1 = topk_labels[0] if topk_labels else "unknown"
    return {"top1": top1, "scores": scores, "raw": {"logits": logits.cpu(), "probs": probs.cpu()}}

# -------------------------
# Fusion helpers
# -------------------------
_SIMPLE_VERBS = ["fry","frying","grill","grilling","bake","baking","boil","boiling",
                 "chop","chopping","slice","slicing","cut","cutting","mix","mixing",
                 "stir","stirring","mince","mincing","grind","grinding","saute","sauteing",
                 "roast","roasting","steam","steaming"]

def pick_verb(label: str) -> str:
    low = label.lower()
    for v in _SIMPLE_VERBS:
        if v in low:
            return v.rstrip("ing") if v.endswith("ing") else v
    for token in low.split():
        if token.endswith("ing"):
            return token[:-3]
    return ""

def pick_nouns_from_photo(photo_label: str) -> str:
    bad = set(_SIMPLE_VERBS + ["with", "and", ",", "in", "on", "a", "the"])
    tokens = [t.strip(",") for t in photo_label.lower().split()]
    tokens = [t for t in tokens if t not in bad and not t.endswith("ing")]
    if not tokens:
        return photo_label
    return " ".join(tokens[-2:])

def fuse_outputs(video_out: Dict[str, Any], photo_outs: List[Dict[str, Any]]):
    video_label = video_out.get("top1", "")
    photo_labels = [p.get("top1", "") for p in photo_outs if p.get("top1")]
    photo_raws = [p.get("raw") for p in photo_outs]
    photo_label = ""
    if photo_labels:
        from collections import Counter
        cnt = Counter(photo_labels)
        photo_label = cnt.most_common(1)[0][0]

    verb = pick_verb(video_label)
    nouns = pick_nouns_from_photo(photo_label)
    generated = f"{verb} {nouns}" if verb and nouns else ""
    report = {"video_top1": video_label, "photo_top1": photo_label, "photo_all_frames": photo_labels}
    return {"generated": generated, "report": report, "video_raw": video_out.get("raw"), "photo_raws": photo_raws}

# -------------------------
# Main orchestration
# -------------------------
def run_inference(input_path: str, run_photo_on_frames: int = 5, topk_video: int = 5):
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)
    logger.info(f"[bold green]Input:[/bold green] {input_path}")

    if is_image(input_path):
        logger.info("Detected image -> running photo model only")
        photo_res = run_photo_inference_wrapper(input_path)
        return {
            "type": "image",
            "photo": photo_res,
            "fusion": {"generated": "", "report": {"video_top1": "", "photo_top1": photo_res["top1"]}}
        }

    if is_video(input_path):
        logger.info("Detected video -> running video model and photo model on frames")
        video_res = run_video_inference_wrapper(input_path, topk=topk_video)
        frame_files = extract_frames_to_files(input_path, num=run_photo_on_frames)
        photo_results = []
        try:
            for f in frame_files:
                logger.info(f"Running photo model on frame {os.path.basename(f)}")
                r = run_photo_inference_wrapper(f)
                photo_results.append(r)
        finally:
            if frame_files:
                tmpdir = os.path.dirname(frame_files[0])
                try:
                    shutil.rmtree(tmpdir)
                    logger.info(f"Removed temp frames dir {tmpdir}")
                except Exception:
                    pass
        fusion = fuse_outputs(video_res, photo_results)
        return {
            "type": "video",
            "video": video_res,
            "photo_frames": photo_results,
            "fusion": fusion
        }
    raise ValueError("Unsupported file type")

# -------------------------
# CLI
# -------------------------
def cli():
    parser = argparse.ArgumentParser(description="Unified inference for b-chef (photo + video + fusion)")
    parser.add_argument("--input", "-i", required=True, help="Path to image or video")
    parser.add_argument("--frames", "-f", type=int, default=5, help="Number of frames to run photo model on")
    parser.add_argument("--topk", "-k", type=int, default=5, help="Top-k for video model")
    args = parser.parse_args()

    res = run_inference(args.input, run_photo_on_frames=args.frames, topk_video=args.topk)
    print("\n[bold]RESULT[/bold]")
    print(res)

if __name__ == "__main__":
    cli()
