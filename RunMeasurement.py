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
abs_path = "S:\\masterproject\\images\\dodecane_17012025\\set2"

# If the config files do not exist, create new ones.
isLive = False
if isLive:
    fps = 1e5
    iim.create_config_from_suite(abs_path, fps)
    
# Load the files and directories from path
contents = os.listdir(abs_path)

# For each measurement, compute the initial radius of the droplets
for directory in contents:
    if bool(re.search("meas", directory)):
        with open(os.path.join(abs_path, directory,'config.yaml')) as file:
            try:
                data = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
            
            R = data['INITIAL_PARAMETERS']['DROP_RADIUS']
            frame_start_top = data['INITIAL_PARAMETERS']['INITIAL_FRAME_TOP_VIEW']
            frame_start_side = data['INITIAL_PARAMETERS']['INITIAL_FRAME_SIDE_VIEW']
            alpha_top = data['MEASUREMENTS_PARAMETERS']['CONV_TOP_VIEW']
            alpha_side = data['MEASUREMENTS_PARAMETERS']['CONV_SIDE_VIEW']
            fps = data['MEASUREMENTS_PARAMETERS']['FPS']
            
            input_dir = os.path.join(abs_path, directory, 'side_view')
            cd = ComputeLensDynamics(input_dir, 
                                     XMIN=0, XMAX=None, 
                                     YMIN=0, YMAX=None, 
                                     framestart=('stack 1', frame_start_side), 
                                     view='side')
            r_max_side = cd.get_R()
            
            input_dir = os.path.join(abs_path, directory, 'top_view')
            cd = ComputeLensDynamics(input_dir, 
                                      XMIN=0, XMAX=None, 
                                      YMIN=0, YMAX=None, 
                                      framestart=('stack 1', frame_start_top), 
                                      view='top')
            r_max_top = cd.get_R()
            
            break
            
            
#%% Plot powerlaw
def power_law(A, x, p):
    return A * (x/x[0])**(p)

plt.close('all')

x_ana = np.linspace(0, 1/fps, len(r_max_side))
r_plot = np.zeros((len(r_max_side), 2))
for ii, r in enumerate(r_max_side):
    temp = []
    for maxima in r:
        temp.append([maxima[0], maxima[1]])
    temp = np.mean(temp, axis=0)
    r_plot[ii] = temp

r_plot = np.linalg.norm(r_plot - r_plot[0], axis=1)        

plt.figure()
plt.loglog(x_ana, r_plot, '.-', color='blue', lw=2)
plt.show()

for ii, r in enumerate(r_max_top):
    if r == []:
        length = ii
        break

x_ana = np.linspace(0, 1/fps, length)
r_plot = np.zeros((length, 2))
for ii in range(0, length):
    r = r_max_top[ii]
    temp = []
    for maxima in r:
        temp.append([maxima[0], maxima[1]])
    temp = np.mean(temp, axis=0)
    r_plot[ii] = temp

r_plot = np.linalg.norm(r_plot - r_plot[0], axis=1)     

plt.figure()
plt.loglog(x_ana, r_plot, '.-', color='blue', lw=2)

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