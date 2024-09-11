# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:10:02 2024

@author: coena
"""
import os
import numpy as np
from tqdm import tqdm
import DetectDroplets as dp
import HelperFunctions as hp
import matplotlib.pyplot as plt

plt.close('all')

path = "C://Users//coena//OneDrive - University of Twente//universiteit//master//master_project//WORK//"
path2 = "C:\\Users\\coena\\OneDrive - University of Twente\\universiteit\\master\\master_project\\WORK\\images\\test_videos\\coalescence_95mPas_video7_20240909_151353"
path3 = ""

# hp.print_directory_tree(path, indent_level=3)
image_paths, _ = hp.load_files(path2)
image = hp.read_from_stack(image_paths[0], 160)

plt.figure()
plt.imshow(image, cmap='gray')


#%%
# Determine window of interest
XMIN = 400
XMAX = 600
YMIN = 250
YMAX = 600

# -----------------------------------------------------------------------------
# ---------------------------- LOAD THE IMAGES --------------------------------
# -----------------------------------------------------------------------------
# hp.print_directory_tree(path, indent_level=3)
image_paths, _ = hp.load_files(path2)

# Create a vector of the starting and ending frame of each tiff stack
frames = []
for path in image_paths:
    frames.append([0, hp.get_stack_size(path)])
    

# -----------------------------------------------------------------------------
# -------------------------- FIND CONNECTING FRAME ----------------------------
# -----------------------------------------------------------------------------
# Get the initial frame where the droplets connect
stack_cnt = 0
frame_cnt = 0
for image_path, entries in zip(image_paths, frames):
    for entry in range(entries[1]):
        # Load image from tif stack
        image = hp.read_from_stack(image_path, entry)[XMIN:XMAX+1, YMIN:YMAX+1]
        
        # Break out of the loop if connection is found
        connected = dp.is_connected(image)
        if connected:
            stack_start = stack_cnt
            frame_start = frame_cnt
            break
        else:
            frame_cnt += 1
            
    # Make sure to break out of the entire loop
    if connected:
        break
    else:
        stack_cnt += 1

# Set the starting frame, at the appropriate stack
frame_end = frames[stack_cnt][1]
frames[stack_cnt] = [frame_start, frame_end]

# -----------------------------------------------------------------------------
# -------------------------- EXTRACT THE EDGES --------------------------------
# -----------------------------------------------------------------------------
# Run through all the tiff stacks, works for just one stack too
stack_tracker = 1
diff = 10
r_max = []
for image_path, entries in zip(image_paths[stack_cnt:], frames[stack_cnt:]):
    print(f"stack {stack_tracker}/{len(image_paths)}")    
    iterator = range(entries[0], entries[1])
    
    # Run through the tiff stack
    for ii, entry in tqdm(enumerate(iterator)):
        # Get image from stack
        
        image = hp.read_from_stack(image_path, entry)[XMIN:XMAX+1, YMIN:YMAX+1]
        
        # Detects edges with subpixel accuracy
        coords_subpix = dp.detect_edges(image, kernel=(5,5))
        
        # Get the maximum from the coordinates (by fitting a spline)
        x_max, y_max = dp.find_edge_extrema(coords_subpix)
        
        # Append the maxima
        r_max.append([x_max, y_max])
        
        # Check if the maximum hasn't started moving
        if ((np.abs(x_max - r_max[ii - 1][0]) > diff) & (ii != 0)):
            r_max.pop()
            print('\nIteration halted at frame ' + str(entry))
            break
        
    # Iterate the stack tracker
    stack_tracker += 1

#%% Plot the maxima
plt.close('all')

def power_law(A, x, p):
    return A * (x/x[0])**(p)

fps = 20E3

r_max_vector = np.zeros((len(r_max), 2))
x_experimental = np.arange(0, r_max_vector.shape[0]) / fps * 1E3
x_analytical = np.geomspace(2, 5, 20)

for ii, r in enumerate(r_max):
    r_max_vector[ii] = r

r_x_diff = np.diff(r_max_vector[:, 0])

plt.figure()
plt.plot(r_x_diff, '.-')
plt.xlabel('t (ms)')
plt.ylabel(r'\Delta x (t)')
plt.grid()

plt.figure()
plt.title(r'Time evolution of $h_0$, mineral oil 95 $mPa\cdot s,\, Oh\approx 0.6$')
plt.loglog(np.abs(r_max_vector[:, 1] - r_max_vector[0, 1]), x_experimental, '.-', label='experimental data')
plt.loglog(x_analytical, power_law(1e-1, x_analytical, 1), '--', color='black', label=r'$\sim t$')
plt.legend()
plt.xlabel('t (ms)')
plt.ylabel('h(t) - h(0) (px)')
plt.grid()



    
