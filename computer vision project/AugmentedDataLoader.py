#!/usr/bin/env python
# coding: utf-8

# In[6]:


import DataLoader
import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt

# Load image dataset
def load_image_dataset(dataset_path):
    train_ds, val_ds = DataLoader.load_image_dataset(dataset_path)
    return train_ds, val_ds

# Data augmentation layers
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1), 
    layers.RandomZoom(height_factor=0.4, width_factor=0.4),
]

# Data augmentation function
def apply_data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

# Visualization of augmented images
def visualize_augmented_images(train_ds):
    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = apply_data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(augmented_images[0]).astype("uint8"))
            plt.axis("off")