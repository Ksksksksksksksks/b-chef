import os
import zipfile
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.utils.data import random_split
import dvc.api
from PIL import Image
import numpy as np
from collections import defaultdict  # Added import for defaultdict
import evaluate

# === Configuration ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data/processed/images")
MODEL_NAME = "dwililiya/food101-model-classification"
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMG_SIZE = 224
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models/fine_tuned_food_model")
NUM_CLASSES = 14  # Your custom classes (meat, eggs, grains)
PUSH_TO_HUB = False  # Set to True to push to Hugging Face (requires login)

# === Data Augmentations (to avoid overfitting) ===
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),  # Random zoom/crop
    transforms.RandomRotation(15),  # Rotation up to 15 degrees
    transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color adjustments
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5),  # Perspective warp
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# === Custom Dataset Class with DVC ===
class DVCFoodDataset(Dataset):
    def __init__(self, zip_paths, max_images_per_category=500, transform=None):
        self.transform = transform
        self.max_images = max_images_per_category
        self.category_to_images = defaultdict(list)  # Now properly defined
        self.classes = set()

        for zip_path in zip_paths:
            with dvc.api.open(zip_path, mode='rb') as zip_file:
                with zipfile.ZipFile(zip_file) as zf:
                    for file_info in zf.infolist():
                        if file_info.filename.endswith('.jpg') and '/' in file_info.filename:
                            category = file_info.filename.split('/')[0]
                            if category:  # Ensure it's a category folder
                                self.classes.add(category)
                                if len(self.category_to_images[category]) < max_images_per_category:
                                    self.category_to_images[category].append(file_info)

        self.classes = sorted(list(self.classes))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return sum(min(len(imgs), self.max_images) for imgs in self.category_to_images.values())

    def __getitem__(self, idx):
        # Map idx to category and image index
        cumulative_size = 0
        for category, images in self.category_to_images.items():
            category_size = min(len(images), self.max_images)
            if idx < cumulative_size + category_size:
                img_idx = idx - cumulative_size
                file_info = images[img_idx]
                break
            cumulative_size += category_size

        with dvc.api.open(os.path.join(DATA_DIR, f"{os.path.basename(file_info.filename).split('.')[0]}.zip.dvc")) as zip_file:
            with zipfile.ZipFile(zip_file) as zf:
                with zf.open(file_info) as img_file:
                    image = Image.open(img_file).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    label = self.class_to_idx[category]
                    return image, label

# === Load and Split Dataset ===
zip_paths = [
    os.path.join(DATA_DIR, "filtered_food_dataset.zip"),
    os.path.join(DATA_DIR, "uec_food256_dataset.zip")
]
full_dataset = DVCFoodDataset(zip_paths, max_images_per_category=500, transform=None)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"Dataset loaded: {len(train_dataset)} train samples, {len(val_dataset)} val samples")
print(f"Classes: {full_dataset.classes}")

# === Load Model ===
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)

# === Training Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# Metrics
accuracy_metric = evaluate.load("accuracy")

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    acc = accuracy_metric.compute(predictions=all_preds, references=all_labels)['accuracy']
    return running_loss / len(loader), acc

def val_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_metric.compute(predictions=all_preds, references=all_labels)['accuracy']
    return running_loss / len(loader), acc

# === Training Loop ===
best_val_acc = 0.0
for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
    scheduler.step(val_loss)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
        print(f"New best model saved with Val Acc: {val_acc:.4f}")

print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")

# === Save Model ===
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

if PUSH_TO_HUB:
    model.push_to_hub("your-username/fine-tuned-custom-food-model")
    processor.push_to_hub("your-username/fine-tuned-custom-food-model")

# === Inference Example ===
def load_model(model_path=OUTPUT_DIR):
    model = AutoModelForImageClassification.from_pretrained(model_path, num_labels=NUM_CLASSES)
    processor = AutoImageProcessor.from_pretrained(model_path)
    model.eval()
    return model, processor