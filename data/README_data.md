
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
# To fetch processed clips or models
dvc pull -r yandex_video_datasets

# To upload newly generated clips or models
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

Once the data is placed, run:

```bash
python src/preprocess/prep_videos_from_zip.py
```

The script will:

1. Read the annotation CSV with action labels and frame ranges.

2. Locate only the required .avi files inside the ZIP archive — it does not unpack the whole dataset.

3. Extract each required video temporarily, then cut it into short clips corresponding to actions (e.g., peel, pour, cut dice).

4. Save each clip temporarily as .avi, add it to DVC, and delete the local .avi to keep storage clean.

Example console output:

```
Processing video s08-d02-cam-002.avi
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

---


### 📋 4. Notes

* The local `.avi` files are **deleted automatically** after being added to DVC.
* Processing large video archives can take several hours.
* Each action label (e.g., `peel`, `cut`, `pour`) has its own subfolder in `video_clips/`.
* You can track preprocessing progress via logs printed to the console.

---
