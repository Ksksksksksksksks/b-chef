# Inference Scripts for Cooking Action and Food Recognition

## Overview

This directory contains the production-ready inference scripts used by the Telegram bot to analyze user-submitted photos and videos.

- **photo_inference.py** → Food recognition, doneness estimation, and container detection (single image).
- **video_inference.py** → Cooking action recognition on short video clips (SlowFast + LoRA).
- **unified_inference.py** → The main script used in the bot: automatically detects image/video, runs the appropriate model(s), extracts frames when needed, and fuses video action + photo food recognition into a clean cooking description (e.g. "fry chicken", "boil egg").

These scripts are designed to run efficiently on both CPU or GPU and will be integrated into the live bot.

## Quick Testing

### 1. Test Photo Inference

```
python inference/photo_inference.py "test_samples/steak_medium.jpg"
```

Expected output:

```
{
  "food": "filet_mignon",
  "doneness": "medium",
  "container": "pan",
  "recommendation": "Detected filet_mignon that is medium on pan. Continue cooking as needed."
}
```

### 2. Test Video Inference (standalone)
```
python inference/video_inference.py "test_samples/frying_egg.mp4" --topk 5
```
This will show the top-5 actions with probabilities and the final averaged prediction.

### 3. Test Unified Inference (recommended)
```
python inference/unified_inference.py --input test_samples/chopping_onion.mp4 --frames 5 --topk 5
# Video (runs both video model + photo model on 5 frames)
```
Example output for a video of someone frying chicken:

```
{
  "type": "video",
  "video": {
    "top1": "frying_something",
    "scores": {"frying_something": 0.987, "stirring": 0.008, ...}
  },
  "photo_frames": [
    {"top1": "fried_chicken"},
    {"top1": "chicken_wings"},
    ...
  ],
  "fusion": {
    "generated": "fry chicken",
    "report": {
      "video_top1": "frying_something",
      "photo_top1": "fried_chicken",
      "photo_all_frames": ["fried_chicken", "chicken_wings", ...]
    }
  }
}
```

The bot uses the fusion.generated field ("fry chicken") as the detected cooking step.

### Key Features & Fixes Already Implemented
Overconfidence guard in unified_inference.py: if the video model predicts one class with ≥99% confidence (common failure mode), it automatically falls back to the next most likely class.

Lazy loading of heavy models (Grounding DINO, CLIP, SlowFast) — only loads what's needed.

Robust frame extraction with fallback (even works on corrupted/short videos).

