# Script 1: Define mappings, fine-tune the model, and save everything needed

import json
import requests
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoImageProcessor, AutoModelForImageClassification
from io import BytesIO
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import DefaultDataCollator
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, GaussianBlur, Compose

# General food categories and their doneness levels
food_doneness_map = {
    'steak': ['raw steak', 'rare steak', 'medium rare steak', 'medium steak', 'medium well steak', 'well done steak'],
    'chicken': ['raw chicken', 'undercooked chicken', 'cooked chicken', 'overcooked chicken'],
    'fish': ['raw fish', 'undercooked fish', 'cooked fish', 'overcooked fish'],
    'eggs': ['raw eggs', 'soft boiled eggs', 'hard boiled eggs', 'fried eggs', 'scrambled eggs', 'overcooked eggs'],
    'vegetables': ['raw vegetables', 'steamed vegetables', 'roasted vegetables', 'fried vegetables', 'overcooked vegetables'],
}

# Container classes
container_classes = "frying pan . plate . bowl . cutting board . oven tray . pot"

# Mapping from food-101 labels to general categories
food101_to_general = {
    'steak': 'steak',
    'filet_mignon': 'steak',
    'prime_rib': 'steak',
    'grilled_salmon': 'fish',
    'fish_and_chips': 'fish',
    'tuna_tartare': 'fish',
    'sashimi': 'fish',
    'chicken_curry': 'chicken',
    'chicken_quesadilla': 'chicken',
    'chicken_wings': 'chicken',
    'fried_egg': 'eggs',
    'deviled_eggs': 'eggs',
    'eggs_benedict': 'eggs',
    'omelette': 'eggs',
    'beet_salad': 'vegetables',
    'caesar_salad': 'vegetables',
    'caprese_salad': 'vegetables',
    'greek_salad': 'vegetables',
}

# Load the base food classification model to fine-tune
food_processor = AutoImageProcessor.from_pretrained("eslamxm/vit-base-food101")
food_model = AutoModelForImageClassification.from_pretrained("eslamxm/vit-base-food101")

# Load Food-101 dataset
dataset = load_dataset("food101")

# For demo, use a small subset to speed up (remove .shuffle().select() for full dataset)
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(2000))
dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(500))

# Define augmentations for training (includes Gaussian blur to handle noisy/blurry inputs)
train_augment = Compose([
    RandomResizedCrop((food_processor.size['height'], food_processor.size['width'])),
    RandomHorizontalFlip(),
    GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
])

# Preprocess function for train (with augmentations)
def train_preprocess(examples):
    images = [train_augment(img.convert("RGB")) for img in examples["image"]]
    inputs = food_processor(images, return_tensors="pt")
    examples["pixel_values"] = inputs["pixel_values"]
    examples["labels"] = examples["label"]
    return examples

# Preprocess function for eval (no augmentations)
def eval_preprocess(examples):
    images = [img.convert("RGB") for img in examples["image"]]
    inputs = food_processor(images, return_tensors="pt")
    examples["pixel_values"] = inputs["pixel_values"]
    examples["labels"] = examples["label"]
    return examples

# Apply preprocessing
dataset["train"] = dataset["train"].map(train_preprocess, batched=True, batch_size=16)
dataset["validation"] = dataset["validation"].map(eval_preprocess, batched=True, batch_size=16)

# Remove unnecessary columns
dataset = dataset.remove_columns(["image"])

# Define compute_metrics for accuracy (we'll add full classification report later)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# Training arguments with label smoothing to reduce noisy data impact
training_args = TrainingArguments(
    output_dir="photo-model_scripts/food_finetune",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",  # Changed from evaluation_strategy
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    label_smoothing_factor=0.1,  # Key: Reduces impact of noisy labels
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",  # Disable logging to external services
)

# Initialize Trainer
trainer = Trainer(
    model=food_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=DefaultDataCollator(return_tensors="pt"),
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Explicitly save the best model (trainer already loads the best at end)
trainer.save_model("photo-model_scripts/food_finetune/best_model")
food_processor.save_pretrained("photo-model_scripts/food_finetune/best_model")

# Save mappings to JSON
with open("photo-model_scripts/food_finetune/mappings.json", "w") as f:
    json.dump({
        "food_doneness_map": food_doneness_map,
        "container_classes": container_classes,
        "food101_to_general": food101_to_general
    }, f)