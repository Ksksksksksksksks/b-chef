import json
import requests
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoImageProcessor, AutoModelForImageClassification
from io import BytesIO
import sys

from pathlib import Path

# Load the fine-tuned food model and processor from the saved directory (with use_fast=True to fix warning)
food_processor = AutoImageProcessor.from_pretrained("models_weights/food_finetune/best_model", use_fast=True)
food_model = AutoModelForImageClassification.from_pretrained("models_weights/food_finetune/best_model")

# Load other pre-trained photo-model_scripts (unchanged)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")

# Load mappings from JSON
with open("models_weights/food_finetune/mappings.json", "r") as f:
    mappings = json.load(f)
    food_doneness_map = mappings["food_doneness_map"]
    container_classes = mappings["container_classes"]
    food101_to_general = mappings["food101_to_general"]

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
    #print(f"Debug: Predicted Food-101 label: {food101_label}, General: {general_food}")  # For debugging precision issues
    return general_food, food101_label

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
    #print(f"Debug: Detection scores for '{prompt}': {results['scores']}")  # Debug low scores if detection fails

    if len(results["boxes"]) == 0:
        #print("Debug: No detection, using original image")
        return image

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
    texts = [f"a close-up photo of {level} in a kitchen setting" for level in doneness_levels]  # Improved prompts for better CLIP accuracy
    inputs = clip_processor(text=texts, images=cropped_image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    predicted_id = probs.argmax().item()
    #print(f"Debug: Doneness probs for {food}: {probs.tolist()}")  # Debug to see confidence
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
    #print(f"Debug: Container detection scores: {results['scores']}")  # Debug

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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        #print("Usage: python vit_usage.py path/to/image")
        sys.exit(1)
    image_path = sys.argv[1]
    result = process_image(image_path)
    print(result)