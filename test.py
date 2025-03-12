# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 11:46:59 2025

@author: coena
"""


import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Input and output directories
input_folder = "D:\\masterproject\\example_figures\\raw_data\\starting_frame"
output_folder = "D:\\masterproject\\example_figures\\raw_data\\starting_frame\\cropped"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

top_left_loc = [250, 70]
width = 100
height = 250
bottom_left_loc = [top_left_loc[0] + width, top_left_loc[1]+ height]

# Define crop settings (left, upper, right, lower)
crop_box = (*top_left_loc, *bottom_left_loc)  # Adjust as needed


# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png", ".jpeg", ".tif")):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        # Apply the crop
        cropped_img = img.crop(crop_box)

        # Save the cropped image
        cropped_img.save(os.path.join(output_folder, filename))

print("Cropping complete for all images!")
