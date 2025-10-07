import os
import zipfile
import requests
from datasets import load_dataset, concatenate_datasets
import pandas as pd

# Compute project root (since script runs from src/preprocess)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Increase download timeout for large files
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "100"

# Load the Food101 dataset
food101 = load_dataset("ethz/food101", split="train")

# Define labels from Food101 (meat, eggs, and grains)
food_labels = {
    "meat": ["chicken_wings", "filet_mignon", "hamburger", "steak", "grilled_salmon", "prime_rib"],
    "eggs": ["deviled_eggs", "eggs_benedict", "huevos_rancheros", "omelette"],
    "grains": ["fried_rice", "macaroni_and_cheese", "spaghetti_bolognese", "spaghetti_carbonara"]
}

# Normalized selected class names (replace '_' with ' ' for matching across datasets)
selected_classes = [food.replace('_', ' ') for foods in food_labels.values() for food in foods]

# Function to filter Food101
def filter_food101(food101, food_labels):
    label_feature = food101.features["label"]
    selected_int_labels = [label_feature.str2int(food) for foods in food_labels.values() for food in foods]
    filtered_food = food101.filter(lambda example: example["label"] in selected_int_labels)
    # Add category string
    filtered_food = filtered_food.map(lambda ex: {"category": label_feature.int2str(ex["label"])})
    return filtered_food

# Download and extract UEC FOOD 256 if not present
food256_zip_path = os.path.join(project_root, "data/raw/food256.zip")
food256_extract_path = os.path.join(project_root, "data/raw/food256")
uecfood256_path = os.path.join(food256_extract_path, "UECFOOD256")
category_txt_path = os.path.join(uecfood256_path, "category.txt")

if not os.path.exists(category_txt_path):
    if not os.path.exists(food256_zip_path):
        print("Downloading UEC FOOD 256 dataset (~3GB)...")
        url = "http://foodcam.mobi/dataset256.zip"
        try:
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()
            os.makedirs(os.path.dirname(food256_zip_path), exist_ok=True)
            with open(food256_zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Download failed: {e}. Please download http://foodcam.mobi/dataset256.zip manually and place it in {os.path.dirname(food256_zip_path)}.")
            exit(1)
    else:
        print("ZIP file found, attempting extraction...")

    print("Extracting UEC FOOD 256...")
    try:
        os.makedirs(food256_extract_path, exist_ok=True)
        with zipfile.ZipFile(food256_zip_path, 'r') as zip_ref:
            zip_ref.extractall(food256_extract_path)
        print("Extraction complete.")
    except zipfile.BadZipFile:
        print(f"Corrupted ZIP file at {food256_zip_path}. Please re-download or replace it.")
        exit(1)
    except Exception as e:
        print(f"Extraction failed: {e}. Check permissions or disk space.")
        exit(1)

    if not os.path.exists(category_txt_path):
        print(f"Error: category.txt not found in {uecfood256_path} after extraction. Verify the ZIP contents.")
        exit(1)

# Parse category.txt from UEC FOOD 256
with open(category_txt_path, "r") as f:
    lines = f.readlines()
    class_id_to_name = {}
    for line in lines[1:]:  # Skip header
        if line.strip():
            parts = line.strip().split()
            class_id = int(parts[0])
            class_name = ' '.join(parts[1:])
            class_id_to_name[class_id - 1] = class_name  # imagefolder