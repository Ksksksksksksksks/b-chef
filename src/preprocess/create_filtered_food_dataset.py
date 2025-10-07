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
food101_zip_path = os.path.join(output_dir, "filtered_food_dataset.zip")
food101_dvc_path = food101_zip_path + ".dvc"

# Config (limit to 500 images per category)
MAX_IMAGES_PER_CATEGORY = 500

# Track how many images we've processed per category
category_counts = defaultdict(int)

# Create ZIP with filtered Food101 images
if os.path.exists(food101_zip_path) or os.path.exists(food101_dvc_path):
    print(f"Skipping {os.path.basename(food101_zip_path)} (already exists)")
else:
    print(f"Processing {os.path.basename(food101_zip_path)}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Organize Food101 images by category
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

        # Create ZIP for Food101
        with zipfile.ZipFile(food101_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for category, paths in category_dirs.items():
                for path in paths:
                    zf.write(path, os.path.join(category, os.path.basename(path)))

        # Add to DVC and remove local ZIP
        try:
            subprocess.run(["dvc", "add", food101_zip_path], check=True)
            os.remove(food101_zip_path)
            print(f"Added to DVC and removed local file: {os.path.basename(food101_zip_path)}\n")
        except subprocess.CalledProcessError as e:
            print(f"Error adding {os.path.basename(food101_zip_path)} to DVC: {e}\n")
            if os.path.exists(food101_zip_path):
                os.remove(food101_zip_path)

# Download and process UEC FOOD 256
uec_zip_path = os.path.join(project_root, "data/raw/food256.zip")
if not os.path.exists(uec_zip_path):
    print("Downloading UEC FOOD 256 (~3GB)...")
    url = "http://foodcam.mobi/dataset256.zip"
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        os.makedirs(os.path.dirname(uec_zip_path), exist_ok=True)
        with open(uec_zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}. Please download http://foodcam.mobi/dataset256.zip manually and place it in {os.path.dirname(uec_zip_path)}.")
        exit(1)

uec_zip_output = os.path.join(output_dir, "uec_food256_dataset.zip")
uec_dvc_path = uec_zip_output + ".dvc"

if os.path.exists(uec_zip_output) or os.path.exists(uec_dvc_path):
    print(f"Skipping {os.path.basename(uec_zip_output)} (already exists)")
else:
    print(f"Processing {os.path.basename(uec_zip_output)}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Parse category.txt from ZIP
        with zipfile.ZipFile(uec_zip_path, 'r') as zf:
            category_txt = zf.read("UECFOOD256/category.txt").decode('utf-8').splitlines()
            class_id_to_name = {int(line.split()[0]): ' '.join(line.split()[1:]) for line in category_txt[1:] if line.strip()}

            # Define the specific folders to process
            selected_folders = {1, 9, 27, 40, 45, 58, 61, 155}
            relevant_categories = {name.replace(' ', '_') for name in class_id_to_name.values() if name in [food.replace('_', ' ') for foods in food_labels.values() for food in foods]}

            # Process images from selected folders
            category_dirs = defaultdict(list)
            for class_id, name in class_id_to_name.items():
                if class_id not in selected_folders or name not in [food.replace('_', ' ') for foods in food_labels.values() for food in foods]:
                    continue
                category = name.replace(' ', '_')
                if category_counts[category] >= MAX_IMAGES_PER_CATEGORY:
                    continue

                folder = str(class_id)
                files = [n for n in zf.namelist() if n.startswith(f"UECFOOD256/{folder}/") and n.endswith('.jpg')]
                files = files[:MAX_IMAGES_PER_CATEGORY - category_counts[category]]  # Limit remaining

                for file in files:
                    basename = os.path.basename(file)
                    out_name = f"uec_{basename}"
                    out_path = os.path.join(tmp_dir, out_name)

                    with zf.open(file) as img_file, open(out_path, "wb") as f:
                        f.write(img_file.read())
                    category_dirs[category].append(out_path)
                    category_counts[category] += 1

            # Create ZIP for UEC FOOD 256
            with zipfile.ZipFile(uec_zip_output, 'w', zipfile.ZIP_DEFLATED) as zf:
                for category, paths in category_dirs.items():
                    for path in paths:
                        zf.write(path, os.path.join(category, os.path.basename(path)))

            # Add to DVC and remove local ZIP
            try:
                subprocess.run(["dvc", "add", uec_zip_output], check=True)
                os.remove(uec_zip_output)
                print(f"Added to DVC and removed local file: {os.path.basename(uec_zip_output)}\n")
            except subprocess.CalledProcessError as e:
                print(f"Error adding {os.path.basename(uec_zip_output)} to DVC: {e}\n")
                if os.path.exists(uec_zip_output):
                    os.remove(uec_zip_output)

print("\n=== Summary ===")
for cat, count in category_counts.items():
    print(f"{cat}: {count} images")