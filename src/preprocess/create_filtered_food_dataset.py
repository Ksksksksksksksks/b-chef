import os
import zipfile
import tempfile
import subprocess
from datasets import load_dataset
from collections import defaultdict

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

# Function to filter Food101
def filter_food101(food101, food_labels):
    label_feature = food101.features["label"]
    selected_int_labels = [label_feature.str2int(food) for foods in food_labels.values() for food in foods]
    filtered_food = food101.filter(lambda example: example["label"] in selected_int_labels)
    # Add category string
    filtered_food = filtered_food.map(lambda ex: {"category": label_feature.int2str(ex["label"])})
    return filtered_food

# Apply filtering
filtered_food = filter_food101(food101, food_labels)

# Paths
base_dir = os.path.dirname(__file__)  # src/preprocess
output_dir = os.path.join(project_root, "data/processed/images")
os.makedirs(output_dir, exist_ok=True)
zip_path = os.path.join(output_dir, "filtered_food_dataset.zip")
dvc_path = zip_path + ".dvc"

# Config (limit to 75 images per category, like MAX_CLIPS_PER_CATEGORY)
MAX_IMAGES_PER_CATEGORY = 500

# Track how many images we've processed per category
category_counts = defaultdict(int)

# Create ZIP with filtered images
if os.path.exists(zip_path) or os.path.exists(dvc_path):
    print(f"Skipping {os.path.basename(zip_path)} (already exists)")
else:
    print(f"Processing {os.path.basename(zip_path)}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Organize images by category
        category_dirs = defaultdict(list)
        for i, example in enumerate(filtered_food):
            category = example["category"].replace(' ', '_')
            if category_counts[category] < MAX_IMAGES_PER_CATEGORY:
                img_path = os.path.join(tmp_dir, f"{category}_{i}.jpg")
                try:
                    example["image"].save(img_path)
                    category_dirs[category].append(img_path)
                    category_counts[category] += 1
                except Exception as e:
                    print(f"Skipping image {i} for {category} due to error: {e}")

        # Create ZIP
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for category, paths in category_dirs.items():
                for path in paths:
                    zf.write(path, os.path.join(category, os.path.basename(path)))

        # Add to DVC and remove local ZIP
        try:
            subprocess.run(["dvc", "add", zip_path], check=True)
            os.remove(zip_path)
            print(f"Added to DVC and removed local file: {os.path.basename(zip_path)}\n")
        except subprocess.CalledProcessError as e:
            print(f"Error adding {os.path.basename(zip_path)} to DVC: {e}\n")

print("\n=== Summary ===")
for cat, count in category_counts.items():
    print(f"{cat}: {count} images")