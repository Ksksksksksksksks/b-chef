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

## 📂 Project Structure
```

/b-chef
/bot          # Telegram bot
/cv           # computer vision (food state detection)
/nlp          # text feedback analysis
/rl           # reinforcement learning
/docs         # documentation

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
