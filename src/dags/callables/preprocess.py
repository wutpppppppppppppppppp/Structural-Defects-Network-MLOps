import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from typing import Tuple
import logging
import torch
from torch.utils.data import Subset, random_split, ConcatDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as TF

# Global logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

### Custom Transform ###
class Denoise:
    def __call__(self, img):
        return TF.gaussian_blur(img, kernel_size=3)

### Transform pipelines ###
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        Denoise(),
        transforms.ToTensor(),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        Denoise(),
        transforms.ToTensor(),
    ])
    logger.info("✅ Transforms initialized.")
    return train_transform, val_test_transform

### Load dataset with error checking ###
def load_combined_dataset(transform) -> Tuple[ConcatDataset, list]:
    categories = ['Decks', 'Walls', 'Pavements']
    root = '/home/santitham/airflow/dags/Structural-Defects-Network-MLOps/Dataset'
    datasets_list = []

    for cat in categories:
        path = os.path.join(root, cat)
        if not os.path.isdir(path):
            logger.error(f"❌ Dataset folder not found: {path}")
            raise FileNotFoundError(f"Dataset folder not found: {path}")

        try:
            ds = datasets.ImageFolder(path, transform=transform)
            if len(ds) == 0:
                logger.warning(f"⚠️ No images found in {path}")
            datasets_list.append(ds)
            logger.info(f"✅ Loaded {cat} dataset with {len(ds)} samples.")
        except Exception as e:
            logger.exception(f"❌ Failed to load dataset from {path}: {e}")
            raise

    if not datasets_list:
        raise RuntimeError("❌ No datasets loaded. Please check directory paths and contents.")
    return ConcatDataset(datasets_list), datasets_list[0].classes

### Dataset split ###
def split_dataset(dataset):
    total_len = len(dataset)
    train_size = int(0.7 * total_len)
    val_size = int(0.15 * total_len)
    test_size = total_len - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])

### Balancing ###
def balance_dataset(train_set):
    logger.info("⚖️ Balancing training dataset...")

    class_0_indices = [i for i, (_, label) in enumerate(train_set) if label == 0]
    class_1_indices = [i for i, (_, label) in enumerate(train_set) if label == 1]

    if not class_0_indices or not class_1_indices:
        logger.warning(f"⚠️ One or both classes are empty. Skipping balancing.")
        return train_set

    min_class_size = min(len(class_0_indices), len(class_1_indices))

    balanced_indices = np.concatenate([
        np.random.choice(class_0_indices, min_class_size, replace=False),
        np.random.choice(class_1_indices, min_class_size, replace=False)
    ])
    np.random.shuffle(balanced_indices)

    logger.info(f"✅ Balanced to {min_class_size} samples per class.")
    return Subset(train_set, balanced_indices)

### Save dataset to disk ###
def save_dataset_as_folder(dataset, save_path, split_name, class_names):
    image_folder = os.path.join(save_path, split_name, "images")
    label_csv = os.path.join(save_path, split_name, "labels.csv")
    os.makedirs(image_folder, exist_ok=True)

    data = []
    for idx, (image, label) in tqdm(enumerate(dataset), total=len(dataset), desc=f"Saving {split_name}"):
        filename = f"{split_name}_{idx:05d}.png"
        filepath = os.path.join(image_folder, filename)
        save_image(image, filepath)
        data.append({"filename": filename, "label": class_names[label]})

    pd.DataFrame(data).to_csv(label_csv, index=False)
    logger.info(f"📁 {split_name} saved to {image_folder} with labels in {label_csv}")

### Main pipeline with error catching ###
def preprocess_data():
    try:
        train_transform, val_test_transform = get_transforms()
        # For val/test
        full_dataset, class_names = load_combined_dataset(val_test_transform)
        train_set, val_set, test_set = split_dataset(full_dataset)

        # For training (with augmentations)
        train_set_full, _ = load_combined_dataset(train_transform)
        balanced_train_set = balance_dataset(train_set_full)

        # Save all sets
        save_dataset_as_folder(balanced_train_set, "artifact_folder", "train", class_names)
        save_dataset_as_folder(val_set, "artifact_folder", "val", class_names)
        save_dataset_as_folder(test_set, "artifact_folder", "test", class_names)

        logger.info("✅ Preprocessing pipeline completed successfully.")
    except Exception as e:
        logger.exception("❌ Preprocessing pipeline failed.")

if __name__ == "__main__":
    preprocess_data()
