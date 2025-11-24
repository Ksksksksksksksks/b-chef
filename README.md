# 🧑‍🍳 B Chef

**Smart Kitchen Assistant for Beginners**

B Chef is a **Telegram bot** that helps absolute beginners cook simple meals.  
Users choose a recipe, send photos during cooking, and the bot gives advice.  
It can speak in two tones:
- 👵 Friendly Grandma  
- 🔥 Strict Gordon Ramsay  

> ⚠️ Note: The functionality may change during development. Features listed here are part of the roadmap, not final.

---

## 🚀 Features
- Simple recipe selection (eggs, soup, pasta, etc.)
- Photo analysis (undercooked / cooked / burnt)
- Text feedback understanding ("burnt", "too watery", etc.)
- Two communication styles
- Reinforcement learning (adapts to user feedback)

---

## 📝 Previous Steps

We have completed preprocessing of the photo datasets and set up the initial food state classification model:

* Food State Classification Model

  * Pre-trained on Food-101 dataset
  * Fine-tuning script:
  ```src/models/train.py```
  * Datasets:

```  
    data/processed/images/filtered_food_dataset.zip.dvc
    data/raw/photos.dvc   # doneness dataset
```    
* Video Action Recognition

  * Model: pre-trained SlowFast
  * Dataset: planned to be used via Kaggle, stored in:

```
    data/processed/video_dataset/tensors.7z.dvc
```
## Current Achievements
- **Photo analysis** → recognizes 101+ foods, doneness level (raw → well-done/overcooked), container (pan, pot, plate, etc.)
- **Video analysis** → recognizes 25 cooking actions (frying, boiling, chopping, stirring, etc.)
- **Fusion of photo + video** → understands the exact step, e.g. "fry chicken", "boil egg", "chop onion"
- Real-time feedback ("Your steak is medium-rare → flip it now", "You're burning the eggs!")
- Works with both photos and short videos
---
## Photo Model (Food + Doneness + Container)
- Base: ViT-B/16 fine-tuned on Food-101 + custom doneness dataset
- Doneness detection: CLIP zero-shot on cropped food (improved prompts)
- Cropping & container detection: Grounding DINO
Example output:
```
{'food': 'grilled_salmon', 'doneness': 'overcooked fish', 'container': 'frying pan', 'recommendation': 'Your grilled_salmon looks overcooked fish. Be careful not to overcook further to avoid dryness.'}
```
---
## Video Action Recognition
- Model: SlowFast-R50 + LoRA (25 kitchen actions)
- Fully trained and accurate on real kitchen videos
---
## Unified Inference (What the bot actually uses)
- Automatically detects image/video
- Runs both models when needed
- Fuses results → clean cooking step ("fry chicken", "stir pasta", etc.)
- Includes overconfidence guard and many production fixes
- Script: `inference/unified_inference.py` ← this will be the core of the bot's brain
---
## 📂 Project Structure 
```
/.dvc   # config for DVC storages
/data   # here all data are stored - raw/preprocessed/external
/src   # here main script are stored - fro preprocessing and models training

````

---

## 🛠️ Tech Stack
- Python 3.10+
- [aiogram](https://docs.aiogram.dev) — Telegram bot
- [PyTorch](https://pytorch.org/) — computer vision + RL
- [spaCy](https://spacy.io/) / scikit-learn — NLP
- SQLite / PostgreSQL — user data storage

---

## 📌 Roadmap
- [ ] Milestone 0: Research (datasets, similar apps, UX)
- [ ] Milestone 1: MVP Bot (recipes + two tones)
- [ ] Milestone 2: Computer Vision (food state detection)
- [ ] Milestone 3: NLP (user text analysis)
- [ ] Milestone 4: RL (adaptive style and advice)
- [ ] Milestone 5: Polish (docs, deployment)

---

## ⚡ Quick Start
1. Clone repository:
   ```bash
   git clone https://github.com/<username>/b-chef.git
   cd b-chef
   ```


2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run bot (skeleton only for now):

   ```bash
   python bot/main.py
   ```

---

## 🤝 Contributing

Pull requests and ideas are welcome.
Please open an issue if you have suggestions — let’s cook together 🍳🔥

---

## 📜 License

MIT License

