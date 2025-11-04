import requests
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoImageProcessor, AutoModelForImageClassification
from io import BytesIO

# Load models
# Food classification model fine-tuned on Food-101
food_processor = AutoImageProcessor.from_pretrained("eslamxm/vit-base-food101")
food_model = AutoModelForImageClassification.from_pretrained("eslamxm/vit-base-food101")

# CLIP for doneness (use larger model for better quality)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# GroundingDINO for object detection (base for better quality)
dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")

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

def get_general_food(food101_label):
    return food101_to_general.get(food101_label, 'unknown')

def recognize_food(image):
    """Recognize specific food using fine-tuned model on Food-101."""
    inputs = food_processor(image, return_tensors="pt")
    with torch.no_grad():
        logits = food_model(**inputs).logits
    predicted_id = logits.argmax(-1).item()
    food101_label = food_model.config.id2label[predicted_id]
    general_food = get_general_food(food101_label)
    return general_food, food101_label  # Return both general and specific

def detect_and_crop(image, prompt):
    """Detect object using GroundingDINO and crop the image."""
    inputs = dino_processor(images=image, text=prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = dino_model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs['input_ids'],
        target_sizes=target_sizes,
        threshold=0.35
    )[0]

    if len(results["boxes"]) == 0:
        return image  # No detection, return original

    # Get the box with highest score
    max_score_idx = results["scores"].argmax()
    box = results["boxes"][max_score_idx].cpu().numpy().astype(int)
    cropped = image.crop((box[0], box[1], box[2], box[3]))
    return cropped

def recognize_doneness(cropped_image, food):
    """Recognize doneness using CLIP on cropped image."""
    if food not in food_doneness_map:
        return "unknown"

    doneness_levels = food_doneness_map[food]
    texts = [f"a photo of {level}" for level in doneness_levels]
    inputs = clip_processor(text=texts, images=cropped_image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    predicted_id = probs.argmax().item()
    return doneness_levels[predicted_id]

def recognize_container(image):
    """Recognize container using GroundingDINO."""
    inputs = dino_processor(images=image, text=container_classes, return_tensors="pt")
    with torch.no_grad():
        outputs = dino_model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs['input_ids'],
        target_sizes=target_sizes,
        threshold=0.35
    )[0]

    if len(results["labels"]) == 0:
        return "unknown"

    max_score_idx = results["scores"].argmax()
    container = results["labels"][max_score_idx]
    return container

def generate_recommendation(food, doneness, container):
    """Generate cooking recommendation."""
    if doneness.startswith(('raw', 'undercooked')):
        return f"Your {food} looks {doneness} on {container}. Continue cooking to reach desired doneness."
    elif doneness.startswith(('overcooked', 'well done', 'hard boiled')):
        return f"Your {food} looks {doneness}. Be careful not to overcook further to avoid dryness."
    else:
        return f"Detected {food} that is {doneness} on {container}. Continue cooking as needed."

def process_image(image_path_or_url):
    """Main function to process the image."""
    if image_path_or_url.startswith("http"):
        image = Image.open(BytesIO(requests.get(image_path_or_url).content))
    else:
        image = Image.open(image_path_or_url)

    # Recognize food
    general_food, specific_food = recognize_food(image)
    if general_food == 'unknown':
        return {"food": specific_food, "doneness": "unknown", "container": "unknown", "recommendation": "Unknown food type."}

    # Detect and crop the food item
    cropped_image = detect_and_crop(image, specific_food)  # Use specific for better detection

    # Recognize doneness on cropped
    doneness = recognize_doneness(cropped_image, general_food)

    # Recognize container
    container = recognize_container(image)

    # Generate recommendation
    recommendation = generate_recommendation(specific_food, doneness, container)

    return {
        "food": specific_food,
        "doneness": doneness,
        "container": container,
        "recommendation": recommendation
    }

# Example usage
# result = process_image("path/to/image.jpg")
# print(result)

"""### Fine-Tuning the Food Classification Model with Label Smoothing to Reduce Noisy Data Impact

Here, we fine-tune the ViT model on the Food-101 dataset. Label smoothing (factor=0.1) is enabled to handle potential noisy labels in the data. We also add simple data augmentations for robustness to noisy inputs (e.g., blur).
"""

from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import DefaultDataCollator
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, GaussianBlur, Compose

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
    output_dir="./food_finetune",
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

history = trainer.state.log_history

epochs = [entry['epoch'] for entry in history if 'loss' in entry]
train_loss = [entry['loss'] for entry in history if 'loss' in entry]
val_loss = [entry['eval_loss'] for entry in history if 'eval_loss' in entry]
train_acc = [entry['train_accuracy'] for entry in history if 'train_accuracy' in entry]  # If tracked; else use eval
val_acc = [entry['eval_accuracy'] for entry in history if 'eval_accuracy' in entry]

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label='loss', color='blue')
plt.plot(epochs, val_loss, label='val_loss', color='green')
if train_acc:
    plt.plot(epochs, train_acc, label='binary_accuracy', color='orange')
plt.plot(epochs, val_acc, label='val_binary_accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.title('Training and Validation Metrics')
plt.legend()
plt.show()

predictions = trainer.predict(dataset["validation"])
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids
target_names = [food_model.config.id2label[i] for i in range(len(food_model.config.id2label))]

report = classification_report(labels, preds, target_names=target_names, digits=3)
print("Classification Report of the Model Trained on Food-101:")
print(report)
