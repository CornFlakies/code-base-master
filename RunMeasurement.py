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
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
from ComputeLensDynamics import ComputeLensDynamics


#%% 
plt.close('all')

abs_path = "D:\\masterproject\\images\\mineral_oil_27032025_94mpas\\set2_zoom"
abs_path = "D:\\masterproject\\images\\mineral_oil_27032025_94mpas\\set2_zoom_higher_fps"
abs_path = "D:\\masterproject\\images\\mineral_oil_27032025_94mpas\\set2_zoom_highest_fps"
abs_path = "D:\\masterproject\\images\\mineral_oil_27032025_94mpas\\set1_no_zoom"
abs_path = "S:\\masterproject\\images\\height_measurements\\11042024\\set6\\"
fps = 1e5

# If the config files do not exist, create new ones.
isLive = True
if isLive:
    # FPS needs to be defined by the user
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
            
            # Load the rest of the measurement data
            frame_start_top = data['INITIAL_PARAMETERS']['INITIAL_FRAME_TOP_VIEW']
            frame_start_side = data['INITIAL_PARAMETERS']['INITIAL_FRAME_SIDE_VIEW']
            R = data['MEASUREMENTS_PARAMETERS']['DROP_RADIUS']
            alpha_top = data['MEASUREMENTS_PARAMETERS']['CONV_TOP_VIEW']
            alpha_side = data['MEASUREMENTS_PARAMETERS']['CONV_SIDE_VIEW']
            fps = data['MEASUREMENTS_PARAMETERS']['FPS']

            # Get drop radii
            all_radii.append(R)
            
            try:                
                # Pre-load data from the manually selected points
                r_max_side = np.load(os.path.join(abs_path, directory, 'manual_points_side.npy'))

                # Compute side view heights
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
            
            except:
                frames_side = 0
                r_max_side = 0
        
            try:
                # Pre load top view points
                r_max_top = np.load(os.path.join(abs_path, directory, 'manual_points_top.npy'))
                
                # Compute top view heights
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
            
            except:
                frames_top = 0
                r_max_top = 0
            
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

# Save to pickle file
file = os.path.join(abs_path, 'data.pkl')
df.to_pickle(file)

# Save to CSV
file = os.path.join(abs_path, 'data.csv')
df.to_csv(file)

