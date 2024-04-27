#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt


# In[2]:


def load_image_dataset(directory, image_size=(224, 224), batch_size=10, validation_split=0.2, seed=1337):
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        directory,
        validation_split=validation_split,
        subset="both",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size
    )
    return train_ds, val_ds


# In[3]:


def show_images(dataset):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
    plt.show()