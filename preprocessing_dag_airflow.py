from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import random
import pickle
from typing import Tuple
from collections import Counter

import numpy as np
import pandas as pd
import cv2

import torch
from torch.utils.data import Subset, random_split, ConcatDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from tqdm import tqdm

# 🛠️ Base Path Setup
base_path = "/home/santitham/airflow/dags/Structural-Defects-Network-MLOps"
TRAIN_IMAGES_PATH = os.path.join(base_path, "artifact_folder", "train", "images")
TRAIN_LABELS_PATH = os.path.join(base_path, "artifact_folder", "train", "labels.csv")
VAL_IMAGES_PATH = os.path.join(base_path, "artifact_folder", "val", "images")
VAL_LABELS_PATH = os.path.join(base_path, "artifact_folder", "val", "labels.csv")
import traceback

LOG_PATH = os.path.join(base_path, "preprocessing", "log.txt")

def log_exception(e):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(f"\n--- Error at {datetime.now()} ---\n")
        f.write(traceback.format_exc())
        f.write("\n")
        
def set_seed():
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

def get_transforms():
    try:
        class Denoise:
            def __call__(self, img):
                return TF.gaussian_blur(img, kernel_size=3)

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

        return train_transform, val_test_transform
    except Exception as e:
        log_exception(e)
        raise e

def load_combined_dataset(transform):
    root = os.path.join(base_path, 'Dataset')
    categories = ['decks', 'pavements', 'walls']
    datasets_list = []
    for category in categories:
        path = os.path.join(root, category)
        dataset = datasets.ImageFolder(path, transform=transform)
        datasets_list.append(dataset)
    return ConcatDataset(datasets_list), datasets_list[0].classes

def split_dataset():
    _, val_test_transform = get_transforms()
    full_dataset, class_names = load_combined_dataset(val_test_transform)

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

    with open(os.path.join(base_path, "artifact_folder", "train_val_test.pkl"), "wb") as f:
        pickle.dump((train_set, val_set, test_set, class_names), f)

def balance_and_save():
    set_seed()
    train_transform, _ = get_transforms()
    train_set_full, class_names = load_combined_dataset(train_transform)

    class_0_indices = [i for i, (_, label) in enumerate(train_set_full) if label == 0]
    class_1_indices = [i for i, (_, label) in enumerate(train_set_full) if label == 1]

    min_class_size = min(len(class_0_indices), len(class_1_indices))
    balanced_indices = np.concatenate([
        np.random.choice(class_0_indices, min_class_size, replace=False),
        np.random.choice(class_1_indices, min_class_size, replace=False),
    ])
    np.random.shuffle(balanced_indices)
    balanced_train_set = Subset(train_set_full, balanced_indices)

    with open(os.path.join(base_path, "artifact_folder", "train_balanced.pkl"), "wb") as f:
        pickle.dump((balanced_train_set, class_names), f)

def save_dataset_as_folder():
    from_path = os.path.join(base_path, "artifact_folder")
    with open(os.path.join(from_path, "train_val_test.pkl"), "rb") as f:
        train_set, val_set, test_set, class_names = pickle.load(f)
    with open(os.path.join(from_path, "train_balanced.pkl"), "rb") as f:
        balanced_train_set, _ = pickle.load(f)

    def _save(dataset, split_name):
        image_folder = os.path.join(from_path, split_name, "images")
        label_csv = os.path.join(from_path, split_name, "labels.csv")
        os.makedirs(image_folder, exist_ok=True)
        data = []
        for idx, (image, label) in tqdm(enumerate(dataset), total=len(dataset), desc=f"Saving {split_name}"):
            filename = f"{split_name}_{idx:05d}.png"
            filepath = os.path.join(image_folder, filename)
            save_image(image, filepath)
            data.append({"filename": filename, "label": class_names[label]})
        pd.DataFrame(data).to_csv(label_csv, index=False)

    _save(balanced_train_set, "train")
    _save(val_set, "val")
    _save(test_set, "test")

def create_dag():
    with DAG(
        dag_id='preprocessing_pipeline_dag',
        start_date=datetime(2024, 1, 1),
        schedule_interval=None,
        catchup=False,
        tags=['preprocessing', 'ml']
    ) as dag:

        task_set_seed = PythonOperator(
            task_id='set_seed',
            python_callable=set_seed
        )

        task_split = PythonOperator(
            task_id='split_dataset',
            python_callable=split_dataset
        )

        task_balance = PythonOperator(
            task_id='balance_dataset',
            python_callable=balance_and_save
        )

        task_save = PythonOperator(
            task_id='save_to_folder',
            python_callable=save_dataset_as_folder
        )

        task_set_seed >> task_split >> task_balance >> task_save

    return dag

globals()['preprocessing_pipeline_dag'] = create_dag()
