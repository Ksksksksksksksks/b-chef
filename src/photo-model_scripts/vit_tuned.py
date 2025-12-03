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
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, GaussianBlur, Compose, CenterCrop, Resize, ToTensor, Normalize
from peft import LoraConfig, get_peft_model
import evaluate

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

# Load Food-101 dataset
dataset = load_dataset("food101")

dataset["train"] = dataset["train"].shuffle(seed=42).select(range(2000))
dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(500))

# Data prepapator for a model
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Extract parameters from image_processor
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

train_transforms = Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

# Map labels from string to int and vice versa (Food-101 has 101 classes)
label2id, id2label = dict(), dict()
labels = dataset["train"].features["label"].names
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

# Use the existing train/validation split from Food-101
train_ds = dataset["train"]
val_ds = dataset["validation"]

train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the base model
model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True
)
model.to(device)
print_trainable_parameters(model)

# Load LoRA config
config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)

lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)

batch_size = 32

args = TrainingArguments(
    "models_weights/food_finetune",
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-3,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    fp16=(device == "cuda"),
    num_train_epochs=50,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    label_smoothing_factor=0.1,  # Kept for noisy data handling
    push_to_hub=False,
    label_names=["labels"],
    report_to='none'
)

metric = evaluate.load("accuracy")

# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# Initialize Trainer
trainer = Trainer(
    model=lora_model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

# Fine-tune the model
print("Starting training...")
trainer.train()

# Explicitly save the best model (trainer already loads the best at end)
trainer.save_model("models_weights/food_finetune/best_model")
image_processor.save_pretrained("models_weights/food_finetune/best_model")

# Save mappings to JSON
with open("models_weights/food_finetune/mappings.json", "w") as f:
    json.dump({
        "food_doneness_map": food_doneness_map,
        "container_classes": container_classes,
        "food101_to_general": food101_to_general
    }, f)
