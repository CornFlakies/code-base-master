# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:35:01 2024

@author: coena
"""

import os
import re
import yaml
import numpy as np
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

# If the config files do not exist, create new ones.
isLive = False
if isLive:
    fps = 1e5
    iim.create_config_from_suite(abs_path, fps)
    
# Load the files and directories from path
contents = os.listdir(abs_path)

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

    
#%% Plot powerlaw
def power_law(height, x_start, x_end, p):
    x = np.linspace(x_start, x_end, 2)
    return x, height * (x/x[0])**(p)

plt.close('all')

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

unit = 1e6

for r_max_top, frames_top, r_max_side, frames_side, R in zip(R_all_top, all_frames_top, R_all_side, all_frames_side, all_radii):

    x_top = (frames_top - frames_top[0]) / fps
    r_plot = np.linalg.norm(r_max_top - r_max_top[0], axis=1)        
    
    ax1.loglog(x_top[1:] * 1e3, r_plot[1:] * alpha_top * unit, '.', color='blue', lw=2)
    ax1.loglog(*power_law(10**(2.2), 10**(-0.5), 10**(0), 1/2), '--', color='red', lw=1)
    # ax1.set_xlim([1e-5, 1e-3])
    ax1.set_ylim([10**(1.7), 10**(2.8)])
    ax1.set_xlabel(f'$t - t_0 [ms]$')
    ax1.set_ylabel(f'$R_0 [\mu m]$')
    ax1.grid()
    
    x_side = (frames_side - frames_side[0]) / fps
    r_plot = np.linalg.norm(r_max_side - r_max_side[0], axis=1)   

    ax2.loglog(x_side[1:] * 1e3, r_plot[1:] * alpha_side * unit, '.', color='blue', lw=2)
    ax2.loglog(*power_law(10**(1), 10**(-1), 10**(0), 2/3), '--', color='red', lw=1)  
    ax2.set_xlabel(f'$t - t_0 [ms]$')
    ax2.set_ylabel(f'$y_0 [\mu m]$')
    ax2.grid()
    
    

plt.show()
# plt.axhline(xmin=int(R), linestyle='--', color='red')
# plt.axvline(frame_start_side / fps, '--', color='red')

# plt.figure()
# plt.loglog(x_ana[1:], power_law(p1_max[7], x_ana[1:], 1/2), '--', color='grey', label=r'$r \sim t^{1/2}$')
# plt.loglog(x_ana[1:], power_law(p1_max[1], x_ana[1:], 1), '--', color='black', label=r'$r \sim t$')
# plt.loglog(x_ana, p1_max, 'o', color='blue', label='data')
# plt.title('Neck radius as a function of time')
# plt.legend()
# plt.ylim([10**(0.5), 1e3])
# plt.xlim([1e1, 1e4])
# plt.xlabel(r'$t\, [\mu s]$')
# plt.ylabel(r'$h_0(t)\, [\mu m]$')
# plt.figure()
# plt.loglog(x_ana[1:], power_law(p2_max[7], x_ana[1:], 1/2), '--', color='grey', label=r'$r \sim t^{1/2}$')
# plt.loglog(x_ana[1:], power_law(p2_max[1], x_ana[1:], 1), '--', color='black', label=r'$r \sim t$')
# plt.loglog(x_ana, p2_max, 'o', color='blue', label='data')
# plt.title('Neck radius as a function of time')
# plt.legend()
# plt.ylim([10**(0.5), 1e3])
# plt.xlim([1e1, 1e4])
# plt.xlabel(r'$t\, [\mu s]$')
# plt.ylabel(r'$h_0(t)\, [\mu m]$')