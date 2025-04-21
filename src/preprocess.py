import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_data(slice=5):
    data_root = 'Dataset/'

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Collect datasets from each category folder
    all_data = []
    categories = ['decks', 'pavements', 'walls']

    for category in categories:
        path = os.path.join(data_root, category)
        dataset = datasets.ImageFolder(root=path, transform=transform)
        all_data.append(dataset)

    # Combine all datasets
    combined_data = torch.utils.data.ConcatDataset(all_data)

    # Create a subset using slicing
    sub_dataset = torch.utils.data.Subset(combined_data, indices=range(0, len(combined_data), slice))

    # All ImageFolder datasets share the same classes, so we can get them from one
    class_names = all_data[0].classes

    return sub_dataset, class_names

# Load dataset
image_dataset, class_names = get_data(slice=1)

# Print total number of images
print("Number of images:", 25000)
print("Class names:", class_names)

# Example: check label of first image
image, label = image_dataset[0]
print("Label index:", label)
print("Label name:", class_names[label])


import os
import torch
import pickle
from torchvision import datasets, transforms
from torch.utils.data import random_split, ConcatDataset
import torchvision.transforms.functional as TF
from typing import Tuple
from torchvision.utils import save_image
from tqdm import tqdm

# Denoising transform
class Denoise:
    def __call__(self, img):
        return TF.gaussian_blur(img, kernel_size=3)     # Apply Gaussian blur for denoising

# Transformations
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),          # Resize to 256x256
    transforms.RandomHorizontalFlip(),      # Randomly flip the image horizontally
    transforms.RandomRotation(15),          # Randomly rotate the image by 15 degrees
    transforms.ColorJitter(brightness=0.2, 
                           contrast=0.2),   # Randomly change brightness and contrast
    Denoise(),
    transforms.ToTensor(),                  # Convert image to tensor (scales to [0, 1])
])

val_test_transform = transforms.Compose([
    transforms.Resize((256, 256)),          # Resize to 256x256
    Denoise(),
    transforms.ToTensor(),
])

# Load combined dataset
def load_combined_dataset(transform) -> Tuple[ConcatDataset, list]:
    root = 'Dataset'
    categories = ['decks', 'pavements', 'walls']
    datasets_list = []

    for category in categories:
        path = os.path.join(root, category)
        dataset = datasets.ImageFolder(path, transform=transform)
        datasets_list.append(dataset)

    return ConcatDataset(datasets_list), datasets_list[0].classes

# Load dataset with test transform (neutral)
full_dataset, class_names = load_combined_dataset(val_test_transform)

# Split 70% train, 15% val, 15% test
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

def save_dataset_as_folder(dataset, save_path, split_name, class_names):
    image_folder = os.path.join(save_path, split_name, "images")
    label_csv = os.path.join(save_path, split_name, "labels.csv")
    os.makedirs(image_folder, exist_ok=True)

    data = []

    for idx, (image, label) in tqdm(enumerate(dataset), total=len(dataset), desc=f"Saving {split_name}"):
        filename = f"{split_name}_{idx:05d}.png"
        filepath = os.path.join(image_folder, filename)
        save_image(image, filepath)  # Save image
        data.append({"filename": filename, "label": class_names[label]})

    df = pd.DataFrame(data)
    df.to_csv(label_csv, index=False)
    print(f"✅ {split_name} saved to {image_folder} with labels in {label_csv}")

save_dataset_as_folder(train_set, "artifact_folder", "train", class_names)
save_dataset_as_folder(val_set, "artifact_folder", "val", class_names)
save_dataset_as_folder(test_set, "artifact_folder", "test", class_names)