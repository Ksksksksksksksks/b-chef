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

## 📝 Current Status

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

  * Model: pre-trained SlowFast (training not started yet)
  * Dataset: planned to be used via Kaggle, stored in:

```
    data/processed/video_dataset/tensors.7z.dvc
```
This reflects the current stage: completed preprocessing for photos and initial setup for model training, while video model preparation is in progress.

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
