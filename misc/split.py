import json
import random
import shutil
from pathlib import Path

# ======================
# PARAMETRY
# ======================
DATASET_DIR = Path("data/")
IMAGES_DIR = DATASET_DIR / "rgb"
DSM_DIR = DATASET_DIR / "dsm"
SAR_DIR = DATASET_DIR/ "sar"
ANNOTATIONS_DIR = Path("track2/annotations")

INPUT_ANN = ANNOTATIONS_DIR / "buildings_only_train.json"
TRAIN_ANN = ANNOTATIONS_DIR / "train.json"
VAL_ANN = ANNOTATIONS_DIR / "val.json"

TRAIN_IMG_DIR = DATASET_DIR / "train" / "rgb"
TRAIN_SAR_DIR = DATASET_DIR / "train" /"sar" 
TRAIN_DSM_DIR = DATASET_DIR / "train" / "dsm"
VAL_IMG_DIR = DATASET_DIR / "val" / "rgb"
VAL_SAR_DIR = DATASET_DIR / "val" / "sar"
VAL_DSM_DIR = DATASET_DIR / "val" / "dsm"

VAL_RATIO = 0.2
RANDOM_SEED = 82

# ======================
# LOAD COCO
# ======================
random.seed(RANDOM_SEED)

with open(INPUT_ANN, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

# ======================
# SPLIT IMAGES
# ======================
image_ids = [img["id"] for img in images]
random.shuffle(image_ids)

val_size = int(len(image_ids) * VAL_RATIO)
val_image_ids = set(image_ids[:val_size])
train_image_ids = set(image_ids[val_size:])

train_images = [img for img in images if img["id"] in train_image_ids]
val_images = [img for img in images if img["id"] in val_image_ids]

train_annotations = [
    ann for ann in annotations if ann["image_id"] in train_image_ids
]
val_annotations = [
    ann for ann in annotations if ann["image_id"] in val_image_ids
]

# ======================
# SAVE ANNOTATIONS
# ======================
TRAIN_ANN.parent.mkdir(parents=True, exist_ok=True)

with open(TRAIN_ANN, "w", encoding="utf-8") as f:
    json.dump({
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories
    }, f)

with open(VAL_ANN, "w", encoding="utf-8") as f:
    json.dump({
        "images": val_images,
        "annotations": val_annotations,
        "categories": categories
    }, f)

# ======================
# COPY IMAGES
# ======================
TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_SAR_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DSM_DIR.mkdir(parents=True, exist_ok=True)
VAL_IMG_DIR.mkdir(parents=True, exist_ok=True)
VAL_SAR_DIR.mkdir(parents=True, exist_ok=True)
VAL_DSM_DIR.mkdir(parents=True, exist_ok=True)

def copy_images(image_list, source_dir, target_dir):
    for img in image_list:
        src = source_dir / img["file_name"]
        dst = target_dir / img["file_name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

copy_images(train_images, IMAGES_DIR,TRAIN_IMG_DIR)
copy_images(train_images, SAR_DIR,TRAIN_SAR_DIR)
copy_images(train_images, DSM_DIR,TRAIN_DSM_DIR)
copy_images(val_images, IMAGES_DIR,VAL_IMG_DIR)
copy_images(val_images, SAR_DIR, VAL_SAR_DIR)
copy_images(val_images, DSM_DIR, VAL_DSM_DIR)

print("✅ Podział zakończony")
print(f"Train: {len(train_images)} obrazów")
print(f"Val:   {len(val_images)} obrazów")
