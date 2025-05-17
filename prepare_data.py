import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from PIL import Image

TARGET_SIZE = (224, 224)
DATA_DIR = Path("data/PlantVillage")
OUTPUT_DIR = Path("data_processed")
SEED = 42

def resize_and_save(img_path, output_path):
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize(TARGET_SIZE)
        img.save(output_path)

def prepare_dataset():
    print("Preparing dataset...")

    # Clear existing processed data
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for split in ["train", "val"]:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

    all_images = []
    all_labels = []

    for class_dir in DATA_DIR.iterdir():
        if class_dir.is_dir():
            label = class_dir.name
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG"))
            all_images.extend(images)
            all_labels.extend([label] * len(images))

    class_counts = np.unique(all_labels, return_counts=True)
    print(f"All labels: {[f'{cls}: {count}' for cls, count in zip(class_counts[0], class_counts[1])]}")
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=SEED
    )

    for split, images, labels in tqdm(zip(["train", "val"], [train_imgs, val_imgs], [train_labels, val_labels])):
        print(f"Resizing {len(images)} images for {split} set")
        for img_path, label in tqdm(zip(images, labels), total=len(images)):
            label_dir = OUTPUT_DIR / split / label
            label_dir.mkdir(parents=True, exist_ok=True)
            dest = label_dir / img_path.name
            resize_and_save(img_path, dest)

    print(f"Dataset prepared at {OUTPUT_DIR}")

if __name__ == "__main__":
    prepare_dataset()