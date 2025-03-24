# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 11:46:59 2025

@author: coena
"""

import os
import skimage as sk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import HelperFunctions as hp
from skimage.io import imread, imsave
from PIL import Image

plt.close('all')

def crop_image(image, x1, x2, y1, y2):
    """
    Crops an image based on the provided (x1, x2, y1, y2) parameters.

    :param image: Image array.
    :param x1: Left boundary.
    :param x2: Right boundary.
    :param y1: Upper boundary.
    :param y2: Lower boundary.
    :return: Cropped image array.
    """
    return image[y1:y2, x1:x2]

def select_crop_area(image):
    """
    Opens an interactive figure to select a cropping area by dragging a rectangle.
    Blocks execution until a selection is made and the figure is closed.
    Returns the selected cropping coordinates.
    """
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    coords = []
    selection_done = False
    
    def onselect(eclick, erelease):
        """Callback function to capture selected coordinates."""
        nonlocal coords, selection_done
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        coords = [x1, x2, y1, y2]
        selection_done = True
        plt.close(fig)
    
    rect_selector = RectangleSelector(ax, onselect, interactive=True, useblit=True,
                                      button=[1])
    
    plt.show(block=False)
    
    while not selection_done:
        plt.pause(0.1)
    
    return coords if coords else None

def process_tiff_stack(input_path, output_path):
    """
    Loads a TIFF stack, allows cropping selection, applies cropping to all images,
    and saves the cropped stack as a new TIFF file.
    
    :param input_path: Path to the input TIFF file.
    :param output_path: Path to save the cropped TIFF file.
    """
    stack = imread(input_path)
    if stack.ndim < 3:
        raise ValueError("Input TIFF stack must have multiple frames.")
    
    # Select cropping area using the first frame
    crop_coords = select_crop_area(stack[0])
    if not crop_coords:
        print("No cropping area selected.")
        return
    
    x1, x2, y1, y2 = crop_coords
    
    # Apply cropping to all frames
    cropped_stack = [Image.fromarray(crop_image(frame, x1, x2, y1, y2)) for frame in stack]
    
    # Save new TIFF stack
    cropped_stack[0].save(os.path.join(output_path, "output.tiff"), save_all=True, append_images=cropped_stack[1:])
    print(f"Cropped TIFF stack saved to {output_path}")


# Input and output directories
input_folder = "D:\\masterproject\\example_figures\\raw_data\\starting_frame"
input_folder = "D:\\masterproject\\images\dodecane_20032025\\set1\\meas6\\top_view"
output_folder = "D:\\masterproject\\images\dodecane_20032025\\set1\\meas6\\top_view\\cropped\\"
hp.create_output_dir(output_folder)

plt.close('all')
image_paths, _ = hp.load_files(input_folder)
process_tiff_stack(image_paths[0], output_folder)



