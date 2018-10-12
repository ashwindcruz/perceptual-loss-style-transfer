"""
Set of functions to aid with images. 
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

def format_image(initial_image):
    """
    Convert an image from tf node format to format that is 
    suitable for plt displaying.
    
    This involves ensuring the image is 3D (removing the batch dimension),
    clipping the image to be between 0 and 255, rounding floats to int, and 
    setting the array type to be integers. 
    
    Args:
        initial_image: The original image from the node
    Returns:
        converted_image: Image to be shown by plt
    """
    if np.ndim(initial_image) == 4:
        initial_image = np.squeeze(initial_image, axis=0)
    
    image_clipped = np.clip(initial_image, 0, 255)
    image_rounded = np.rint(image_clipped)
    formatted_image = np.asarray(image_rounded, dtype=np.uint8)
    
    return formatted_image

def display_image(initial_image, fig_title):
    """
    Display an image within the notebook.
    
    Args:
        initial_image: The original image from the node
        fig_title    : Title for this image 
    Returns:
        None
    """
    converted_image = format_image(initial_image)
    plt.figure()
    plt.imshow(converted_image)
    plt.axis('off')
    plt.title(fig_title)

def save_image(initial_image, fig_title, folder_path):
    """
    Save an image to disk.
    
    Args:
        initial_image: The original image from the node
        fig_title    : Title for this image 
        folder_path  : Folder to save the image to
    Returns:
        None
    """
    converted_image = format_image(initial_image)
    img_name = '{}/img_{}.png'.format(folder_path, fig_title)
    plt.imsave(img_name, converted_image)