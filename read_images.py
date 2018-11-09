"""
Given a folder, load up batches of images for the network to use
during training.
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import config as cfg

TRAIN_IMAGE_FILENAMES = tf.gfile.Glob(cfg.TRAIN_IMAGES_DIR + '*')
NUM_TRAIN_IMAGES = len(TRAIN_IMAGE_FILENAMES)

VAL_IMAGE_FILENAMES = tf.gfile.Glob(cfg.VAL_IMAGES_DIR + '*')
NUM_VAL_IMAGES = len(VAL_IMAGE_FILENAMES)

def sample_batch():
    """
    Sample a batch of images for the network to train with.

    Args:
        None
    Returns:
        img_batch: Batch of images
    """
    idx_choices = np.random.choice(num_train_images, cfg.BATCH_SIZE, replace=False)
    chosen_images = np.take(train_image_filenames, idx_choices)

    img_batch = np.empty((cfg.BATCH_SIZE, cfg.HEIGHT, cfg.WIDTH, cfg.CHANNELS))
    for i, image_path in enumerate(chosen_images):
        image = plt.imread(image_path)
        image = cv2.resize(image, (cfg.HEIGHT, cfg.WIDTH))
        if np.ndim(image) == 2:
            image = np.reshape(
                np.repeat(image, cfg.CHANNELS),
                (cfg.HEIGHT, cfg.WIDTH, cfg.CHANNELS))
        image = np.asarray(image, dtype=np.float32)

        img_batch[i, :, :, :] = image

    return img_batch

def val_images():
    """
    Fetch and return the validation images in a batch.

    Args:
        None
    Returns:
        img_batch: Batch of images
    """

    img_batch = np.empty(
        (len(val_image_filenames), cfg.HEIGHT, cfg.WIDTH, cfg.CHANNELS))
    for i, image_path in enumerate(val_image_filenames):
        image = plt.imread(image_path)
        image = cv2.resize(image, (cfg.HEIGHT, cfg.WIDTH))
        image = np.asarray(image, dtype=np.float32)

        img_batch[i, :, :, :] = image

    return img_batch

def fetch_batch(start_index, dataset):
    """
    Fetch a batch of images for training or validation.
    Start at the provided index. 

    Args: 
        start_index: Index to begin batch from.
        dataset    : Choice to take batch from either training or validation set. 
    """
    if dataset == 'train':
        chosen_images = TRAIN_IMAGE_FILENAMES[
            start_index:(start_index+cfg.BATCH_SIZE)]
    else:
        chosen_images = VAL_IMAGE_FILENAMES[
            start_index:(start_index+cfg.BATCH_SIZE)]

    img_batch = np.empty((cfg.BATCH_SIZE, cfg.HEIGHT, cfg.WIDTH, cfg.CHANNELS))
    for i, image_path in enumerate(chosen_images):
        image = plt.imread(image_path)
        image = cv2.resize(image, (cfg.HEIGHT, cfg.WIDTH))
        if np.ndim(image) == 2:
            image = np.reshape(
                np.repeat(image, cfg.CHANNELS),
                (cfg.HEIGHT, cfg.WIDTH, cfg.CHANNELS))
        image = np.asarray(image, dtype=np.float32)

        img_batch[i, :, :, :] = image

    return img_batch