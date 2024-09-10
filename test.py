# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:10:02 2024

@author: coena
"""
import os
from tqdm import tqdm
import DetectDroplets as dp
import HelperFunctions as hp
import matplotlib.pyplot as plt

plt.close('all')

path = "C://Users//coena//OneDrive - University of Twente//universiteit//master//master_project//WORK//"
path2 = "C:\\Users\\coena\\OneDrive - University of Twente\\universiteit\\master\\master_project\\WORK\\images\\test_videos\\coalescence_95mPas_video7_20240909_151353"

# Determine window of interest
framestart = 170
XMIN = 400
XMAX = 600
YMIN = 250
YMAX = 600

# hp.print_directory_tree(path, indent_level=3)
image_paths, _ = hp.load_files(path2)

# Create a list of the amount of entries in each tiff stack
frames = []
for path in image_paths:
    frames.append(hp.get_stack_size(path))

# Run through all the tiff stacks, works for just one stack too
stack_cnt = 1
for image_path, entries in zip(image_paths, frames):
    print(f"stack {stack_cnt}/{len(image_paths)}")
    #Apply the frame start condition
    if stack_cnt == 1:
        iterator = range(framestart, entries)
    else:
        iterator = range(entries)
    
    # Run through the tiff stack
    for entry in tqdm(iterator):
        image = hp.read_from_stack(image_path, entry)[XMIN:XMAX+1, YMIN:YMAX+1]
        # edges_x, edges_y = dp.detect_edges_devernay(image)
        
        
        plt.figure()
        plt.title("Devernay Edge Detection")
        plt.imshow(image)
        plt.scatter(edges_x, edges_y, color="magenta", marker=".", linewidth=.1)
        plt.show()
        break
    stack_cnt += 1 