from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import random
import pickle
from typing import Tuple
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Subset, random_split, ConcatDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as TF

from tqdm import tqdm
import traceback

# DAG default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 5, 4),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'structural_defects_network_pipeline',
    default_args=default_args,
    description='A pipeline for structural defects network training',
    schedule_interval=timedelta(days=1),
    catchup=False,
)

# Base paths
base_path = "/home/santitham/airflow/dags/Structural-Defects-Network-MLOps"
TRAIN_IMAGES_PATH = os.path.join(base_path, "artifact_folder", "train", "images")
TRAIN_LABELS_PATH = os.path.join(base_path, "artifact_folder", "train", "labels.csv")
VAL_IMAGES_PATH = os.path.join(base_path, "artifact_folder", "val", "images")
VAL_LABELS_PATH = os.path.join(base_path, "artifact_folder", "val", "labels.csv")
TEST_IMAGES_PATH = os.path.join(base_path, "artifact_folder", "test", "images")
TEST_LABELS_PATH = os.path.join(base_path, "artifact_folder", "test", "labels.csv")

# Set seeds
def set_seeds(**kwargs):
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    print("Seeds set to:", SEED)
    return SEED

# Custom Denoise transform
class Denoise:
    def __call__(self, img):
        return TF.gaussian_blur(img, kernel_size=3)  # Apply Gaussian blur for denoising


# Define data preparation functions
def get_data(slice=5, **kwargs):
    import logging
    import traceback
    from datetime import datetime
    from airflow.utils.log.logging_mixin import LoggingMixin
    logger = LoggingMixin().log
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(base_path, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a log file specifically for this task
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"get_data_task_{timestamp}.log")
    
    def write_to_log(message):
        with open(log_file, "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    write_to_log("Starting get_data function")
    
    try:
        # Check if base path exists
        write_to_log(f"Base path: {base_path}")
        if not os.path.exists(base_path):
            error_msg = f"Base path does not exist: {base_path}"
            write_to_log(f"ERROR: {error_msg}")
            raise FileNotFoundError(error_msg)
        
        # Check Dataset directory
        data_root = os.path.join(base_path, 'Dataset')
        write_to_log(f"Looking for dataset at: {data_root}")
        if not os.path.exists(data_root):
            error_msg = f"Dataset directory not found: {data_root}"
            write_to_log(f"ERROR: {error_msg}")
            raise FileNotFoundError(error_msg)
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        write_to_log("Created image transform")
        
        # Collect datasets from each category folder
        all_data = []
        # Using directory listing to get actual case of category folders
        categories = os.listdir(data_root)
        write_to_log(f"Found categories: {categories}")
        
        # Filter out any non-directory items
        categories = [c for c in categories if os.path.isdir(os.path.join(data_root, c))]
        write_to_log(f"Directory categories: {categories}")
        
        for category in categories:
            path = os.path.join(data_root, category)
            write_to_log(f"Processing category: {category} at path: {path}")
            
            if not os.path.exists(path):
                error_msg = f"Category directory not found: {path}"
                write_to_log(f"ERROR: {error_msg}")
                raise FileNotFoundError(error_msg)
            
            # List files in directory to verify content
            files = os.listdir(path)
            write_to_log(f"Found {len(files)} files/directories in {path}")
            
            # List first 5 files for debugging
            if files:
                sample_files = files[:5]
                write_to_log(f"Sample files: {sample_files}")
            
            try:
                write_to_log(f"Attempting to create ImageFolder for {path}")
                dataset = datasets.ImageFolder(root=path, transform=transform)
                write_to_log(f"Successfully loaded dataset for {category} with {len(dataset)} images")
                all_data.append(dataset)
            except Exception as e:
                error_msg = f"Error loading dataset for {category}: {str(e)}"
                write_to_log(f"ERROR: {error_msg}")
                # Save full traceback
                tb = traceback.format_exc()
                write_to_log(f"Traceback:\n{tb}")
                raise
        
        # Combine all datasets
        write_to_log(f"Combining {len(all_data)} datasets")
        combined_data = torch.utils.data.ConcatDataset(all_data)
        
        # Create a subset using slicing
        indices = list(range(0, len(combined_data), slice))
        write_to_log(f"Creating subset with slice={slice}, resulting in {len(indices)} images")
        sub_dataset = torch.utils.data.Subset(combined_data, indices=indices)
        
        # All ImageFolder datasets share the same classes, so we can get them from one
        class_names = all_data[0].classes
        
        write_to_log(f"Number of images: {len(sub_dataset)}")
        write_to_log(f"Class names: {class_names}")
        
        # Store only metadata in XCom, not the actual dataset objects
        kwargs['ti'].xcom_push(key='sub_dataset_len', value=len(sub_dataset))
        kwargs['ti'].xcom_push(key='class_names', value=class_names)
        kwargs['ti'].xcom_push(key='dataset_categories', value=categories)
        
        write_to_log("get_data function completed successfully")
        
        # Return metadata only, not the actual dataset objects
        return {
            'dataset_size': len(sub_dataset),
            'class_names': class_names,
            'categories': categories
        }
    
    except Exception as e:
        error_msg = f"Error in get_data function: {str(e)}"
        logger.error(error_msg)
        write_to_log(f"CRITICAL ERROR: {error_msg}")
        
        # Save detailed traceback to file
        tb = traceback.format_exc()
        write_to_log(f"Detailed traceback:\n{tb}")
        
        # Create a dedicated traceback file
        traceback_file = os.path.join(logs_dir, f"get_data_error_{timestamp}.traceback")
        with open(traceback_file, "w") as f:
            f.write(f"Error: {str(e)}\n\n")
            f.write(f"Traceback:\n{tb}")
        
        write_to_log(f"Traceback saved to: {traceback_file}")
        
        # Re-raise the exception for Airflow to handle
        raise

def load_combined_dataset(transform_type='val_test', **kwargs):
    import logging
    import traceback
    import shutil
    from datetime import datetime
    from airflow.utils.log.logging_mixin import LoggingMixin
    from torchvision.datasets import ImageFolder

    logger = LoggingMixin().log

    logs_dir = os.path.join(base_path, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"load_combined_dataset_{timestamp}.log")

    def write_to_log(message):
        with open(log_file, "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    write_to_log("Starting load_combined_dataset function")

    try:
        root = os.path.join(base_path, 'Dataset')
        categories = ['Decks', 'Pavements', 'Walls']
        datasets_list = []

        write_to_log(f"Transform type selected: {transform_type}")

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

        transform = train_transform if transform_type == 'train' else val_test_transform
        write_to_log("Transform initialized")

        for category in categories:
            path = os.path.join(root, category)
            if not os.path.exists(path):
                error_msg = f"Category path not found: {path}"
                write_to_log(f"ERROR: {error_msg}")
                raise FileNotFoundError(error_msg)

            write_to_log(f"Loading dataset from {path}")
            dataset = datasets.ImageFolder(path, transform=transform)
            write_to_log(f"Loaded {len(dataset)} images from {category}")
            datasets_list.append(dataset)

        # Merge all datasets into a new folder for caching
        combined_dir = os.path.join(base_path, "artifact_folder", "full_combined_dataset")
        if os.path.exists(combined_dir):
            shutil.rmtree(combined_dir)

        for dataset in datasets_list:
            for img_path, label in dataset.samples:
                class_name = dataset.classes[label]
                category_name = os.path.basename(os.path.dirname(img_path))  # e.g. Decks, Walls
                target_dir = os.path.join(combined_dir, category_name, class_name)
                os.makedirs(target_dir, exist_ok=True)
                shutil.copy2(img_path, os.path.join(target_dir, os.path.basename(img_path)))

        write_to_log(f"Saved combined dataset to {combined_dir}")

        # For XCom and return purposes
        full_dataset = ConcatDataset(datasets_list)
        class_names = datasets_list[0].classes
        write_to_log(f"Combined dataset size: {len(full_dataset)}")
        write_to_log(f"Class names: {class_names}")

        ti = kwargs.get('ti')
        if ti:
            ti.xcom_push(key='full_dataset_len', value=len(full_dataset))
            ti.xcom_push(key='class_names', value=class_names)
            write_to_log("load_combined_dataset function completed successfully")
        else:
            write_to_log("Warning: 'ti' not found in kwargs. Skipping XCom push.")

        return len(full_dataset), class_names

    except Exception as e:
        error_msg = f"Error in load_combined_dataset function: {str(e)}"
        logger.error(error_msg)
        write_to_log(f"CRITICAL ERROR: {error_msg}")

        tb = traceback.format_exc()
        write_to_log(f"Detailed traceback:\n{tb}")
        traceback_file = os.path.join(logs_dir, f"load_combined_dataset_error_{timestamp}.traceback")
        with open(traceback_file, "w") as f:
            f.write(f"Error: {str(e)}\n\nTraceback:\n{tb}")
        write_to_log(f"Traceback saved to: {traceback_file}")
        raise
    
def load_cached_combined_dataset(transform_type='val_test'):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        Denoise(),
        transforms.ToTensor(),
    ]) if transform_type != 'train' else transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        Denoise(),
        transforms.ToTensor(),
    ])

    dataset_path = os.path.join(base_path, "artifact_folder", "full_combined_dataset")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Cached dataset folder not found: {dataset_path}")

    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    return dataset

def split_dataset(**kwargs):
    ti = kwargs['ti']

    # Pull data from XCom
    full_dataset_len = ti.xcom_pull(task_ids='load_data', key='full_dataset_len')
    class_names = ti.xcom_pull(task_ids='load_data', key='class_names')

    print(f"Total dataset size (from XCom): {full_dataset_len}")
    print(f"Class names (from XCom): {class_names}")

    # Optional: Reconstruct dataset if needed, or use this info only for logging/statistics
    # This assumes re-running `load_combined_dataset` or accessing raw files again if needed

    train_size = int(0.7 * full_dataset_len)
    val_size = int(0.15 * full_dataset_len)
    test_size = full_dataset_len - train_size - val_size

    ti.xcom_push(key='train_size', value=train_size)
    ti.xcom_push(key='val_size', value=val_size)
    ti.xcom_push(key='test_size', value=test_size)

    return train_size, val_size, test_size

def save_dataset_as_folder(dataset, save_path, split_name, class_names, **kwargs):
    """
    Save dataset images to folder and create a labels CSV, with detailed logging.
    """

    logger = LoggingMixin().log

    # Base logs directory
    logs_dir = os.path.join(save_path, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"save_dataset_{split_name}_{timestamp}.log")

    def write_to_log(message):
        with open(log_file, "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    write_to_log(f"Starting save_dataset_as_folder for split: {split_name}")

    try:
        image_folder = os.path.join(save_path, split_name, "images")
        label_csv = os.path.join(save_path, split_name, "labels.csv")
        os.makedirs(image_folder, exist_ok=True)
        write_to_log(f"Created image folder at: {image_folder}")

        data = []

        write_to_log(f"Saving {len(dataset)} images to folder...")
        for idx, (image, label) in tqdm(enumerate(dataset), total=len(dataset), desc=f"Saving {split_name}"):
            filename = f"{split_name}_{idx:05d}.png"
            filepath = os.path.join(image_folder, filename)
            save_image(image, filepath)
            data.append({"filename": filename, "label": class_names[label]})

            if idx < 3:  # Log sample of first few
                write_to_log(f"Saved image: {filename}, label: {class_names[label]}")

        df = pd.DataFrame(data)
        df.to_csv(label_csv, index=False)
        write_to_log(f"Saved label CSV to: {label_csv}")
        write_to_log(f"✅ Successfully saved {len(data)} images and labels for split: {split_name}")

        # XCom push if ti is provided
        ti = kwargs.get("ti", None)
        if ti:
            ti.xcom_push(key=f"{split_name}_saved_count", value=len(data))
        else:
            write_to_log("WARNING: 'ti' not found in kwargs, skipping xcom_push")

        return len(data)

    except Exception as e:
        error_msg = f"Error in save_dataset_as_folder: {str(e)}"
        logger.error(error_msg)
        write_to_log(f"CRITICAL ERROR: {error_msg}")

        tb = traceback.format_exc()
        write_to_log(f"Detailed traceback:\n{tb}")

        # Save traceback to a separate file
        traceback_file = os.path.join(logs_dir, f"save_dataset_error_{split_name}_{timestamp}.traceback")
        with open(traceback_file, "w") as f:
            f.write(f"Error: {str(e)}\n\nTraceback:\n{tb}")

        write_to_log(f"Traceback saved to: {traceback_file}")
        raise

def prepare_train_data(**kwargs):
    """
    Load training data with augmentations and balance classes.
    """
    import os
    import numpy as np
    import logging
    import traceback
    from collections import Counter
    from datetime import datetime
    from airflow.utils.log.logging_mixin import LoggingMixin
    from torch.utils.data import Subset
    
    logger = LoggingMixin().log
    
    # Create logs directory
    logs_dir = os.path.join(base_path, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"prepare_train_data_{timestamp}.log")
    
    def write_to_log(message):
        with open(log_file, "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    write_to_log("Starting prepare_train_data function")

    try:
        write_to_log("Loading combined dataset with training transformations")
        train_set, class_names = load_combined_dataset(transform_type='train')
        write_to_log(f"Loaded dataset with {len(train_set)} samples and classes: {class_names}")

        # Get indices of each class
        write_to_log("Indexing samples by class")
        class_indices = {}
        for i in range(len(class_names)):
            class_indices[i] = [j for j, (_, label) in enumerate(train_set) if label == i]
        write_to_log(f"Class sample counts: {[len(indices) for indices in class_indices.values()]}")

        # Find minimum class size for balancing
        min_class_size = min(len(indices) for indices in class_indices.values())
        write_to_log(f"Minimum class size found: {min_class_size}")

        # Balance dataset by sampling
        balanced_indices = []
        for class_idx, indices in class_indices.items():
            balanced_class_indices = np.random.choice(indices, min_class_size, replace=False)
            balanced_indices.extend(balanced_class_indices)
        write_to_log(f"Balanced indices generated. Total: {len(balanced_indices)}")

        # Shuffle indices
        np.random.shuffle(balanced_indices)
        write_to_log("Shuffled balanced indices")

        # Create balanced dataset
        balanced_train_set = Subset(train_set, balanced_indices)
        write_to_log(f"Balanced dataset created with {len(balanced_train_set)} samples")

        # Verify and log class distribution
        balanced_label_counts = Counter([label for _, label in balanced_train_set])
        write_to_log(f"Balanced class distribution: {dict(balanced_label_counts)}")

        # Save balanced dataset to artifact folder
        artifact_path = os.path.join(base_path, "artifact_folder")
        write_to_log(f"Saving balanced dataset to: {artifact_path}")
        save_dataset_as_folder(balanced_train_set, artifact_path, "train", class_names)

        # Push XCom metadata
        kwargs['ti'].xcom_push(key='balanced_train_set_size', value=len(balanced_train_set))
        kwargs['ti'].xcom_push(key='class_distribution', value=str(dict(balanced_label_counts)))
        write_to_log("XCom push complete")

        write_to_log("prepare_train_data function completed successfully")
        return len(balanced_train_set)

    except Exception as e:
        error_msg = f"Error in prepare_train_data: {str(e)}"
        logger.error(error_msg)
        write_to_log(f"CRITICAL ERROR: {error_msg}")

        tb = traceback.format_exc()
        write_to_log(f"Detailed traceback:\n{tb}")

        # Save traceback to file
        traceback_file = os.path.join(logs_dir, f"prepare_train_data_error_{timestamp}.traceback")
        with open(traceback_file, "w") as f:
            f.write(f"Error: {str(e)}\n\n")
            f.write(f"Traceback:\n{tb}")

        write_to_log(f"Traceback saved to: {traceback_file}")
        raise

def prepare_val_data(**kwargs):
    """Prepare validation data."""
    full_dataset = load_cached_combined_dataset(transform_type='val_test')
    class_names = full_dataset.classes

    # Get split sizes from XCom
    ti = kwargs['ti']
    train_size = ti.xcom_pull(task_ids='split_dataset', key='train_size')
    val_size = ti.xcom_pull(task_ids='split_dataset', key='val_size')
    test_size = ti.xcom_pull(task_ids='split_dataset', key='test_size')

    # Recreate the split
    _, val_set, _ = random_split(full_dataset, [train_size, val_size, test_size])

    # Save validation dataset
    save_dataset_as_folder(val_set, os.path.join(base_path, "artifact_folder"), "val", class_names)

    return len(val_set)

def prepare_test_data(**kwargs):
    """Prepare test data."""
    full_dataset, class_names = load_combined_dataset(transform_type='val_test')
    
    # Get split sizes from XCom
    ti = kwargs['ti']
    train_size = ti.xcom_pull(task_ids='split_dataset', key='train_size')
    val_size = ti.xcom_pull(task_ids='split_dataset', key='val_size')
    test_size = ti.xcom_pull(task_ids='split_dataset', key='test_size')
    
    # Recreate the split
    _, _, test_set = random_split(full_dataset, [train_size, val_size, test_size])
    
    # Save test dataset
    save_dataset_as_folder(test_set, os.path.join(base_path, "artifact_folder"), "test", class_names)
    
    return len(test_set)

def create_dag(dag_id, schedule, default_args):
    dag = DAG(
        dag_id=dag_id,
        default_args=default_args,
        schedule_interval=schedule,
        start_date=datetime(2023, 1, 1),
        catchup=False
    )

    with dag:
        # Create task instances
        set_seeds_task = PythonOperator(
            task_id='set_seeds',
            python_callable=set_seeds,
        )

        get_data_task = PythonOperator(
            task_id='get_data',
            python_callable=get_data,
            op_kwargs={'slice': 1},
        )
        
        load_data_task = PythonOperator(
            task_id='load_data',
            python_callable=load_combined_dataset,
            provide_context=True,
            op_kwargs={'transform_type': 'val_test'},
        )

        split_dataset_task = PythonOperator(
            task_id='split_dataset',
            python_callable=split_dataset,
        )

        prepare_train_data_task = PythonOperator(
            task_id='prepare_train_data',
            python_callable=prepare_train_data,
        )

        prepare_val_data_task = PythonOperator(
            task_id='prepare_val_data',
            python_callable=prepare_val_data,
        )

        prepare_test_data_task = PythonOperator(
            task_id='prepare_test_data',
            python_callable=prepare_test_data,
        )

        # Define task dependencies
        set_seeds_task >> get_data_task >> load_data_task >> split_dataset_task
        split_dataset_task >> prepare_train_data_task
        split_dataset_task >> prepare_val_data_task
        split_dataset_task >> prepare_test_data_task

    return dag

dag_id = 'concrete_structure_preprocessing'
schedule = '@daily'
default_args = {
    'owner': 'airflow',
    'retries': 1,
}

globals()[dag_id] = create_dag(dag_id, schedule, default_args)