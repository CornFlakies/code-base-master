# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:35:01 2024

@author: coena
"""

import os
import re
import yaml
import numpy as np
import pandas as pd
import skimage as sk
from tqdm import tqdm
import InitImage as iim
import HelperFunctions as hp
import matplotlib.pyplot as plt
from ComputeLensDynamics import ComputeLensDynamics

plt.close('all')
    
# Define location measurement suite
abs_path = "D:\\masterproject\\images\\dodecane_17012025\\set2"
# abs_path = "S:\\masterproject\\images\\dodecane_17012025\\set2"

eta = 1.36e-3 # Pa s
gamma =  25e-3 # N / m
rho = 750 # kg / m^3
l_v = eta**2 / (gamma * rho)

#%% 

# If the config files do not exist, create new ones.
isLive = False
if isLive:
    # FPS needs to be defined by the user
    fps = 1e5
    iim.create_config_from_suite(abs_path, fps)
    
# Load the files and directories from path
contents = os.listdir(abs_path)

# Create list to prepare data for pandas dataframe
df_data = []

# Create lists for the r data of each measurement
R_all_side = []
R_all_top = []
all_frames_side = []
all_frames_top = []
all_radii = []

# For each measurement, compute the initial radius of the droplets
for directory in contents:
    if bool(re.search("meas", directory)):
        with open(os.path.join(abs_path, directory,'config.yaml')) as file:
            try:
                data = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
            
            # Pre-load data from the manually selected points
            r_max_side = np.load(os.path.join(abs_path, directory, 'manual_points_side.npy'))
            r_max_top = np.load(os.path.join(abs_path, directory, 'manual_points_top.npy'))
            
            frame_start_top = data['INITIAL_PARAMETERS']['INITIAL_FRAME_TOP_VIEW']
            frame_start_side = data['INITIAL_PARAMETERS']['INITIAL_FRAME_SIDE_VIEW']
            R = data['INITIAL_PARAMETERS']['DROP_RADIUS']
            alpha_top = data['MEASUREMENTS_PARAMETERS']['CONV_TOP_VIEW']
            alpha_side = data['MEASUREMENTS_PARAMETERS']['CONV_SIDE_VIEW']
            fps = data['MEASUREMENTS_PARAMETERS']['FPS']
            
            # Get drop radii
            all_radii.append(R)
            
            # Compute heights
            input_dir = os.path.join(abs_path, directory, 'side_view')
            cd = ComputeLensDynamics(input_dir, 
                                      XMIN=0, XMAX=None, 
                                      YMIN=0, YMAX=None, 
                                      framestart=('stack 1', frame_start_side), 
                                      view='side')
            data_side = cd.get_R()
            
            # add the manually selected data to the computed data, then append
            # to a list containing all the individual measurement data
            for ii, r in enumerate(data_side): # Side view
                if (r != []):
                    r_max_side[frame_start_side + ii, :] = [r[0][0], r[0][1]]
            frames_side = np.where(~np.isnan(r_max_side[:, 0]))[0]
            all_frames_side.append(frames_side)
            r_max_side = r_max_side[~np.isnan(r_max_side).any(axis=1)]
            R_all_side.append(r_max_side)
        
            # Compute heights
            input_dir = os.path.join(abs_path, directory, 'top_view')
            cd = ComputeLensDynamics(input_dir, 
                                      XMIN=0, XMAX=None, 
                                      YMIN=0, YMAX=None, 
                                      framestart=('stack 1', frame_start_top), 
                                      view='top')
            data_top = cd.get_R()
            
            for ii, r in enumerate(data_top): # Top view
                if (r != []):
                    r_max_top[frame_start_top + ii, :] = [r[0][0], r[0][1]]
                    
            # Get indices of all non nan elements
            frames_top = np.where(~np.isnan(r_max_top[:, 0]))[0]
            all_frames_top.append(frames_top)
            r_max_top = r_max_top[~np.isnan(r_max_top).any(axis=1)] 
            R_all_top.append(r_max_top)
            
            # Define data
            data = {"R_max_top": r_max_top * alpha_top,
                    "frames_top": frames_top,
                    "Y_max_side": r_max_side * alpha_side,
                    "frames_side": frames_side,
                    "drop_radii": R,
                    "fps" : fps,
                    "alpha_top": alpha_top,
                    "alpha_side": alpha_side}
            
            # append to list
            df_data.append(data)
            
# Create data frame from dicts
df = pd.DataFrame(df_data) 

# Save to csv file
file = os.path.join(abs_path, 'data.pkl')
df.to_pickle(file)

#%% Plot powerlaw


def find_matching_indices(arr1, arr2):
    i, j = 0, 0  # Two pointers
    matching_indices = []

    while i < len(arr1) and j < len(arr2):
        if arr1[i] == arr2[j]:  # Found a match
            matching_indices.append((i, j))
            i += 1
            j += 1
        elif arr1[i] < arr2[j]:  # Move pointer of smaller value
            i += 1
        else:
            j += 1

    return matching_indices

def power_law(height, x_start, x_end, p):
    x = np.linspace(x_start, x_end, 2)
    return x, height * (x/x[0])**(p)

import matplotlib 
 
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 15}

matplotlib.rc('font', **font)

# Open DF
file = os.path.join(abs_path, 'data.pkl')
df = pd.read_pickle(file)

plt.close('all')

fig1, ax1 = plt.subplots(figsize=(8, 6))
fig2, ax2 = plt.subplots(figsize=(8, 6))
fig3, ax3 = plt.subplots(figsize=(8, 6))
fig4, ax4 = plt.subplots(figsize=(8, 6))
fig5, ax5 = plt.subplots(figsize=(8, 6))

unit = 1e6

for ii in df.index:
    
    # Load all the data from the dataframe
    r_max_top = df.loc[ii, 'R_max_top']
    frames_top = df.loc[ii, 'frames_top']
    r_max_side = df.loc[ii, 'Y_max_side']
    frames_side = df.loc[ii, 'frames_side']
    R = df.loc[ii, 'drop_radii']
    fps = df.loc[ii, 'fps']
    alpha_top = df.loc[ii, 'alpha_top']
    alpha_side = df.loc[ii, 'alpha_side']

    # Define x-values
    x_top = (frames_top - frames_top[0]) / fps
    r_plot_top = np.linalg.norm(r_max_top - r_max_top[0], axis=1)    
    
    # Define inertial time scale
    t_i = np.sqrt((rho * R**3)/gamma)
    
    # plt.figure()
    # plt.plot(r_max_top[:, 0] / alpha_top, r_max_top[:, 1] / alpha_top, '.-', color='green')
    
    # Plot of the top view heights of each individual measurement
    ax1.loglog(x_top[1:] * 1e6, r_plot_top[1:] * unit, '.', color='blue', lw=2)
    # ax1.set_xlim([1e-5, 1e-3])

    x_side = (frames_side - frames_side[0]) / fps
    r_plot_side = np.linalg.norm(r_max_side - r_max_side[0], axis=1)

    # Plot of side view heights of each individual measurement
    ax2.loglog(x_side[1:] * 1e6, r_plot_side[1:] * unit, '.', color='blue', lw=2)
    
    # Plot the top view heights of each individual measurement, divided over
    # the intrinsic length scale
    ax3.loglog(x_top[1:] / t_i, r_plot_top[1:] / R, '.', color='blue', lw=2)
    
    # Plot the top view heights of each individual measurement, divided over
    # the intrinsic length scale
    ax4.loglog(x_side[1:] / t_i, r_plot_side[1:] / l_v, '.', color='blue', lw=2)
    
    # Get h_0 / y_0
    indices = find_matching_indices(frames_top, frames_side)
    f = []
    theta = []
    for jj in indices[1:]:
        y_0 = r_plot_top[jj[0]]
        h_0 = r_plot_side[jj[1]]
        theta.append(h_0 / y_0)
        f_0 = frames_top[jj[0]]
        f.append(f_0)
        
    t = (f - frames_top[0]) / fps   
    ax5.plot(t * 1e3, theta, '.', label=f'{ii}')
    # ax5.loglog(*power_law(1.25e-1, 1e-1, 1e0, 1/6), '--', color='black')
    ax5.set_ylabel("$y_0 / h_0$")
    ax5.set_xlabel("$t [ms]$")
    ax5.set_ylim([0.1, 0.3])
    # ax5.legend()

ax1.loglog(*power_law(10**(2), 10**(2.2), 10**(2.7), 1/2), '--', color='red', lw=2, label='$y_0\sim t^{1/2}$')
ax1.legend()
ax2.loglog(*power_law(10**(1.50), 10**(2.5), 10**(3), 2/3), '--', color='red', lw=2, label='$h_0\sim t^{2/3}$')  
ax2.legend()

ax1.set_title("Top view, $y_0$ as a function of time")
ax1.set_ylim([10**(1.7), 10**(2.8)])
ax1.set_xlabel('$t - t_0 [\mu s]$')
ax1.set_ylabel('$y_0 [\mu m]$')

ax2.set_title("Side view, $h_0$ as a function of time")
ax2.set_xlabel('$t - t_0 [\mu s]$')
ax2.set_ylabel('$h_0 [\mu m]$')

ax3.set_title("top view, divided by the drop radius")
ax3.set_xlabel('$(t - t_0) / t_i$')
ax3.set_ylabel('$y_0\, /\, R$')

ax4.set_title("side view, divided by the intrinsic length scale of dodecane")
ax4.set_xlabel('$(t - t_0) / t_i$')
ax4.set_ylabel('$h_0\, /\, l_v$')

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
ax5.grid()

fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig5.tight_layout()

plt.show()
