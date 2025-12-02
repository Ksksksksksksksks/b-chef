# 🧑‍🍳 B-Chef
**A Multimodal Smart Kitchen Assistant for Beginners**

B-Chef is a **Telegram bot** designed to assist complete beginners in cooking simple meals. Users select basic recipes (e.g., boiling eggs, frying chicken, cooking pasta) and follow step-by-step guidance. They submit photos or short videos (including Telegram video circles) of their progress, and the bot analyzes the input to provide real-time feedback on ingredients, doneness, actions, and corrections. To make interactions engaging, the bot offers three distinct tones:
* 👵 **Friendly Grandma** (warm and encouraging)
* 🔥 **Strict Gordon Ramsay** (direct and no-nonsense)
* 👨‍🍳 **Neutral** (straightforward and factual)

The bot adapts its tone dynamically using a reinforcement learning module based on user reactions (likes/dislikes).

This project was developed as an interdisciplinary course assignment at Innopolis University, combining Practical Machine Learning and Deep Learning (PMLDL), Introduction to Computer Vision (CV), and Reinforcement Learning (RL). It frames the problem of novice cooking challenges, curates and explores datasets, experiments with multiple models, fine-tunes them, and deploys a functional system.

---

# 🚀 Features
* Simple recipe selection focused on beginner tasks (e.g., boiling eggs, frying meat/chicken, cooking rice/pasta)
* Photo analysis: ingredient recognition (14 basic categories), doneness estimation (raw, medium, done, overcooked), container detection (pan, pot, bowl, etc.)
* Video action recognition (25 common kitchen actions, e.g., chopping, stirring, pouring)
* Real-time validation of cooking steps with proactive warnings (e.g., "You're burning it!") and confirmations
* Three communication tones with RL-based adaptation
* Supports photos, videos, and Telegram video circles
* Unified inference pipeline for robust multimodal output fusion
* Rule-based feedback generation with fallback mechanisms for uncertain predictions

---

# 🎯 Current Achievements
### ✔ Research & Problem Framing
* Defined target users: **beginner cooks** (e.g., students or novices) facing challenges like undercooking (health risks), overcooking, or incorrect actions in simple recipes
* Analyzed over **10 existing solutions** including cooking apps (e.g., MimiCook, CookAR), bots, and research papers on CV-assisted cooking
* Conducted competitor analysis to identify gaps in real-time visual feedback for basic tasks
* Explored and compared **multiple datasets**: Food-101, UEC FOOD-256, FoodX-251, iFood-2019, custom doneness datasets for photos; MPII Cooking Activities, YouCook2 for videos
* Researched pre-trained models for transfer learning: ResNet18, ViT-base (for photos); SlowFast, VideoMAE, ViViT (for videos); CLIP, Grounding DINO (for doneness and localization)
* Drafted preliminary system architecture: Telegram bot → unified inference (CV models) → RL tone adaptation → feedback
* Gained insights into domain shifts (e.g., curated datasets vs. real kitchen photos) and the need for robustness in lighting, angles, and user-generated content

### ✔ Data Collection & Preprocessing
* Curated a filtered version of **Food-101** to 14 basic ingredient categories (e.g., chicken, beef/pork, eggs, rice, pasta, buckwheat) with ~7,000 images, supplemented from UEC FOOD-256 for balance
* Assembled a **custom doneness dataset** focusing on states for key items (meat, eggs, fish, vegetables)
* Processed a subset of **MPII Cooking Activities** (~800 clips) into tensor format, focusing on 13 relevant actions + 12 from pre-trained Kinetics-400 heads (total 25 actions)
* Applied augmentations: random resized cropping, horizontal flipping, Gaussian blur for photos; frame extraction and transformations for videos
* Used **DVC** for data versioning and reproducibility
* Built preprocessing scripts for loading, filtering, balancing classes, and handling variations in lighting/texture
* Organized data structure for efficient training, validation, and inference

### ✔ Photo Analysis
* Recognizes 14 basic ingredient categories from images
* Estimates doneness levels (raw, medium, done, overcooked) for key foods (e.g., steak, chicken, fish, eggs, vegetables)
* Detects container types (e.g., pan, pot, bowl) via localization
* Uses **Grounding DINO** for cropping and localizing relevant regions
* Applies **CLIP zero-shot** classification with custom prompts on cropped areas for doneness
* Fine-tuned **ViT-B/16** on filtered Food-101 subset (~2,000 train/500 val samples) with label smoothing (factor=0.1) and augmentations
* Achieved 88% macro F1-score (improved from 84% baseline without augmentations)

### ✔ Video Analysis
* **SlowFast-R50** backbone with **LoRA** adapters for efficient fine-tuning
* Trained on 25 kitchen actions (13 custom from MPII: e.g., cut apart, cut dice, grate, mix, peel; 12 from pre-trained Kinetics-400)
* Handles short videos and Telegram circles, with single-frame fallback for static inputs
* Validation accuracy ~73.68% (early stopping at epoch 19), but addressed overfitting via overconfidence rejection mechanism (filters predictions with unrealistically high confidence)

### ✔ Fusion of Photo + Video
* Combines outputs to generate clear cooking steps, e.g.:
  * "fry chicken"
  * "boil pasta"
  * "chop onion"
* Rule-based logic: action verbs from video + nouns/ingredients from photo
* Includes heuristics for stability, conflict resolution, and low-confidence fallbacks

### ✔ Unified Inference
* Automatically detects input type (photo, video, or circle)
* Routes to appropriate pipeline: Grounding DINO → ViT → CLIP (photos); frame extraction → SlowFast (videos)
* Fusion module produces integrated output with recommendations
* Implements guards against overconfident or unstable predictions
* Central script: `inference/unified_inference.py`

### ✔ RL Module
* **Contextual multi-armed bandit** with decaying epsilon-greedy exploration
* Adapts tone (Grandma, Gordon, Neutral) based on user feedback across cooking sessions
* Maintains and visualizes Q-table for learning progress
* Logs, saves, and updates preferences for personalized interactions

### ✔ End-to-End Bot
* Fully functional Telegram bot using **aiogram** framework
* Integrates all components for real-time conversations
* Guides users through recipes with visual validations, warnings, and tone-adapted advice
* Handles noisy/unclear inputs with fallbacks (e.g., request resubmission)
* Supports complete cooking flows for basic recipes

---

# 🧠 Photo Model Details (Ingredient + Doneness + Container)
* **Backbone:** ViT-B/16
* **Fine-tuning:** On filtered Food-101 subset + custom doneness data, using Hugging Face Trainer API
* **Doneness:** CLIP zero-shot with refined prompts (e.g., "overcooked fish")
* **Localization:** Grounding DINO for cropping food regions and detecting containers
* **Output example:**
```
{'ingredient': 'grilled_salmon',
 'doneness': 'overcooked',
 'container': 'frying pan',
 'recommendation': 'Your grilled salmon looks overcooked. Reduce heat immediately.'}
```

---

# 🎥 Video Action Recognition
* **Model:** SlowFast-R50 optimized with LoRA adapters
* **Action Set:** 25 kitchen actions (13 custom from MPII Cooking videos + 12 pre-trained from Kinetics-400)
* **Pipeline:** Frame extraction → transformations → SlowFast inference
* **Note:** Overfitting mitigated with overconfidence rejection; suitable for real kitchen clips but generalizes moderately to unseen variations

---

# 🧩 Unified Inference
The core engine powering the bot:
* Detects input type automatically
* Selects and runs the appropriate model stack
* Performs fusion to generate actionable cooking steps
* Applies filters for reliability (e.g., reject overconfident wrong predictions)
* Central file: `inference/unified_inference.py`

---

# 📂 Project Structure
```
/.dvc # DVC storage configuration
/data # Raw, processed, and external datasets
/src # Preprocessing, training, and model scripts
/inference # Unified inference pipeline and logic
/model_weights # Final model checkpoints
/rl # Reinforcement learning module and utilities
/bot # Telegram bot implementation
```

---

# 🛠 Tech Stack
* Python 3.10+
* PyTorch – For CV and ML models
* Hugging Face Transformers – ViT, CLIP
* OpenMMLab/MMAction2 – SlowFast
* Grounding DINO – Object detection
* aiogram – Telegram bot framework
* scikit-learn – Prototypes for metrics and RL
* DVC – Data versioning

---

# ✅ What We Have Implemented
* Functional Telegram bot with support for photos, videos, and circles
* Fine-tuned ViT-B/16 on curated Food-101 subset for ingredient recognition
* CLIP-based zero-shot doneness estimation with custom prompts
* Grounding DINO for cropping and container detection
* SlowFast-R50 + LoRA for video action recognition on custom dataset
* Rule-based fusion for generating cooking steps
* Real-time advice with warnings and validations
* Contextual bandit RL for tone adaptation, with Q-table logging/visualization
* Complete end-to-end cooking flows for basic recipes
* DVC-managed datasets and model weights

---

# ❌ What Is Not Implemented
* Full end-to-end fine-tuning of photo model (current version is under-trained on full data)
* Advanced video model (e.g., switch to VideoMAE for better efficiency)
* Dockerized deployment or server-side hosting
* Full NLP module for processing user text complaints (currently template-based)
* Expanded recipe set beyond basics (limited by CV capabilities)
* Natural language generation for dynamic responses (uses fixed templates)

---

# ⚠️ Current Limitations
* Photo model may misclassify under poor lighting or unusual angles, despite augmentations
* CLIP doneness estimation struggles with ambiguous states or non-standard presentations
* Video model occasionally outputs incorrect actions with high confidence (mitigated but not eliminated)
* Fusion can produce inconsistent steps if photo/video inputs conflict
* Inference speed varies by hardware (slower on CPU)
* Recipes are constrained to models' trained categories and actions
* RL adaptation is basic and could benefit from more user data

---

# ⭐ Strengths of the Project
* Multimodal integration: Combines photo, video, and RL for a practical cooking assistant
* Addresses real beginner pain points with visual feedback over text reliance
* Robust engineering: Unified pipeline, fallbacks, and DVC for reproducibility
* Honest experimentation: Explored multiple models, documented challenges like domain shift
* Interactive deployment: Working bot provides engaging, adaptive user experience

---

# 🔮 Future Work & Improvements
### CV & ML Enhancements
* Retrain SlowFast on larger mixed datasets (e.g., full YouCook2) for better generalization
* Explore advanced video models (VideoMAE, TimeSformer, X3D) for efficiency
* Full fine-tuning of ViT to 90%+ accuracy with stronger augmentations
* Integrate more datasets for broader ingredient/doneness coverage

### User Interaction
* Add NLP for parsing text feedback (e.g., "it's too salty")
* Implement natural language generation (e.g., via GPT) for varied responses
* Support dynamic tone blending or additional styles

### Recipes & UX
* Expand to 10–20 recipes with nutritional info (e.g., from USDA FoodData)
* Add "recovery mode" for fixing mistakes (e.g., overcooked food)
* Collect anonymous usage data to refine RL

### Engineering
* Dockerize for easy deployment
* Optimize for server-side inference
* Add monitoring for model drift in production

---

# ⚡ Quick Start
```bash
git clone https://github.com/Ksksksksksksksks/b-chef.git
cd b-chef
pip install -r requirements.txt
# Set up Telegram bot token in .env
python bot/main.py
```

---

# 🤝 Contributing
Contributions, ideas, and improvements are welcome. Open an issue or PR—let's make cooking easier together 🍳🔥

---

# 📜 License
MIT License
