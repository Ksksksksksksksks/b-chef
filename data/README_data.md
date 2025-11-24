
## 🖼️ Photo Data — Preprocessing

This section describes how photo datasets are filtered and prepared for model fine-tuning.

---

### Step 1: Dataset Filtering and Preparation

To adapt open-source datasets (mainly **Food-101**) to the project’s focus on *basic cooking* (e.g., frying meat, boiling eggs), a preprocessing script filters only relevant food classes based on **custom label mappings**.
Filtered images (~7,000 samples) are combined with similar classes from other datasets to maintain balance.

Run:

```bash
python src/preprocess/create_filtered_food_dataset.py
```

This script:

* Loads **Food-101** and **UEC FOOD 256** datasets (downloads UEC FOOD 256 if not present).
* Filters images by relevant categories (meat, eggs, grains).
* Limits the number of images per category (default 500).
* Organizes images in temporary directories and packages them into `.zip` archives.
* Adds the archives to **DVC** and removes local copies to save space.

Output:

```
data/processed/images/
├── filtered_food_dataset.zip.dvc
├── uec_food256_dataset.zip.dvc
```

---

### Step 2: Doneness Dataset

A complementary dataset was created to help detect **cooking stages** (raw, boiled, fried, etc.) for **meat, chicken, eggs, and grains**.
This dataset is used to train the model to recognize **doneness levels** for proteins and grains, which is important for cooking guidance and food safety.

The dataset is stored in **DVC** and can be accessed locally via:

```
data/raw/photos.dvc
```

**Content overview:**

* **Chicken:** raw, boiled, fried
* **Meat:** raw, partially cooked
* **Eggs:** raw, boiled, scrambled, fried
* **Grains:** rice, pasta, buckwheat at different cooking stages

**Usage:**

* Can be loaded by preprocessing scripts to generate train/validation splits.
* Supports model training for doneness classification alongside the main filtered food dataset.

📁 Link: [Doneness Dataset (Google Drive)](https://drive.google.com/drive/folders/10F2WjIbj8SIc5qbZZ0bpM4kJnTf64M98?usp=sharing)

---

### Summary

* Filters and structures raw food images for 14 key categories.
* Applies augmentation and tensor conversion for efficient training.
* Includes a separate dataset for doneness classification.



## 🎥 Video Data — Preprocessing & Storage Guide

This section describes how to prepare and manage the **video data** used in the B-Chef project.

---

### ☁️ 1. Configure Your DVC Remote Storage

The project uses [DVC](https://dvc.org/) to manage large video and model files.
Each developer should configure their own local or cloud storage path before running preprocessing or training scripts.

By default, `.dvc/config` contains several named remotes:

```ini
['remote "yandex_local"']
    url = Z:\b_chef
['remote "yandex_photo_datasets"']
    url = Z:\b_chef\photo_datasets
['remote "yandex_video_datasets"']
    url = Z:\b_chef\video_datasets
['remote "yandex_models"']
    url = Z:\b_chef\models
```

These are just **example local paths** (in this case, Yandex.Disk mounted as drive `Z:`).
You can modify them to match your environment using the command below:

```bash
dvc remote modify yandex_video_datasets url /your/custom/path/video_datasets
```

If you use a cloud storage service (e.g., Yandex Cloud, Google Drive, AWS S3), configure the corresponding remote URL:

```bash
dvc remote modify yandex_video_datasets url s3://mybucket/video_datasets
```

Once your remotes are configured, you can **pull or push** data as usual:

```bash
# To fetch processed clips or photo-model_scripts
dvc pull -r yandex_video_datasets

# To upload newly generated clips or photo-model_scripts
dvc push -r yandex_video_datasets
```

---

### 📦 2. Download the Raw Dataset

The project uses the **MPII Cooking Activities** dataset, which contains high-resolution kitchen videos and corresponding ground-truth annotations.

You need to **manually download** the following files from the official website:
🔗 [MPII Cooking Activities Dataset](https://www.mpii.de/computervision/benchmark/mpiicooking/)

Download:

* `mpii_cooking_videos.zip` — the archive with all raw videos
* `mpii_cooking_groundtruth.csv` — annotations with activity labels and timestamps

Then place them in:

```
data/raw/video/
├── mpii_cooking_videos.zip
└── mpii_cooking_groundtruth.csv
```

---

### ⚙️ 3. Run Preprocessing

#### Step 3.1: Extract and Segment Video Clips

Once the raw data is placed, run the initial preprocessing script:

```bash
python src/preprocess/prep_videos_from_zip.py
```

The script will:

1. Read the annotation CSV with action labels and frame ranges.
2. Read the list of **required action categories** from a text file (`mpii_actions_suitable.txt`), which was manually curated based on:
   * all categories in the dataset, and
   * existing categories in the pre-trained model,
     so that **they do not overlap** with the model’s outputs.
3. Locate only the required `.avi` files inside the ZIP archive — it does not unpack the whole dataset.
4. Extract each required video temporarily, then cut it into short clips corresponding to the selected actions (e.g., peel, pour, cut dice).
5. Clip duration is optionally limited to a maximum number of seconds to prevent overly long videos, which helps training.
6. Save each clip temporarily as `.avi`, add it to DVC, and delete the local `.avi` to keep storage clean.

Example console output:

```
Processing video s08-d02-cam-002.avi, category peel
Added to DVC and removed local file: s08-d02-cam-002_1433_2347.avi
```

After this step, your folder will look like:

```
data/processed/video_clips/
├── peel/
│   ├── s08-d02-cam-002_1433_2347.avi.dvc
│   ├── ...
├── cut/
│   ├── ...
```

Each `.avi.dvc` file is a **lightweight metadata file** tracked by Git, while the actual video data is managed by DVC in remote storage.

#### Step 3.2: Create Train/Val/Test Splits

After the video clips are processed and stored in DVC, run the next script to prepare the data for training:
```bash

python src/preprocess/prep_video_dataset.py
```

This script performs several key operations:

- Splits the dataset: Divides all video_clips into train, validation, and test sets for model fine-tuning.

- Creates metadata CSV files: Generates train.csv, val.csv, and test.csv files that map each clip to its corresponding label.

#### Step 3.3: Precompute Tensors

Since the model is planned to be trained on Kaggle, it was decided to add our custom dataset to Kaggle. To do that, it is needed to generate tensors locally, since local DVC storage are not supported on Kaggle.

To do that, run:
```bash

python src/preprocess/create_video_tensors.py
```
It will create optimized tensor representations from the video clips (found both in DVC remote and DVC cache) and save them in the data/processed/tensors/ directory, organized by split (train/, val/, test/).

The resulting structure after steps 3.2-3.3:
```text

data/processed/
├── video_clips/          # Original clips (via DVC pointers)
├── tensors/              # Precomputed tensors for fast loading
│   ├── train/            # Training tensors
│   ├── val/              # Validation tensors
│   └── test/             # Test tensors
├── train.csv             # Training set metadata
├── val.csv               # Validation set metadata  
└── test.csv              # Test set metadata
```

#### Step 3.4: Prepare for Cloud Training (Kaggle)

For training on Kaggle or other cloud platforms, the tensor files are compressed into a single archive:

```bash
# The tensors are being compressed into 7zip format for Kaggle upload
data/processed/tensors.7z
```

This archive contains the entire `tensors/` directory structure with precomputed tensors, enabling fast data loading during model training without the need for on-the-fly video processing.

---

### 📋 4. Notes

* The local `.avi` files are **deleted automatically** after being added to DVC.
* Processing large video archives can take several hours.
* Each action label (e.g., `peel`, `cut`, `pour`) has its own subfolder in `video_clips/`.
* The precomputed tensors significantly **speed up training** by eliminating video decoding overhead during epochs.
* The train/val/test split ensures proper evaluation and prevents data leakage during model fine-tuning.
* You can track preprocessing progress via logs printed to the console.

