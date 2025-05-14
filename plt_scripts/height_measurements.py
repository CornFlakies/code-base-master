# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 11:16:35 2025

@author: Coen Arents
"""

import os
import re
import yaml
import numpy as np
import pandas as pd
import skimage as sk
from tqdm import tqdm
# import InitImage as iim
import HelperFunctions as hp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
from ComputeLensDynamics import ComputeLensDynamics

fps = 100000
unit = 1e6
H = 0.5e-3 #mm

plt.close('all')
def plot_df_top(df, fig, ax, color='blue', label=None):
    for ii in df.index:
        # Load all the data from the dataframe
        r_max_top = df.loc[ii, 'R_max_top']
        frames_top = df.loc[ii, 'frames_top']
        r_max_side = df.loc[ii, 'Y_max_side']
        frames_side = df.loc[ii, 'frames_side']
        R = df.loc[ii, 'drop_radii']
        # fps = df.loc[ii, 'fps']
        alpha_top = df.loc[ii, 'alpha_top']
        alpha_side = df.loc[ii, 'alpha_side']
        
        # Define x-values and y-values of top view plot
        x_side = (frames_side - frames_side[0]) / fps
        r_plot_side = np.linalg.norm(r_max_side - r_max_side[0], axis=1)    
        
        # Plot of the top view heights of each individual measurement
        ax.loglog(x_side[1:] * unit, r_plot_side[1:] * unit, '.', color=color, label=label)

H = np.array([1.824, 1.6, 2.1, 2.1, 5.2, 50]) * 1e-3
# A colorset
cmap = plt.cm.Blues
norm = mcolors.LogNorm(vmin=min(H)*0.1, vmax=max(H))

abs_path = "S:\\masterproject\\images\\height_measurements\\11042024\\set1\\"
file = os.path.join(abs_path, 'data.pkl')
df = pd.read_pickle(file)
fig1, ax1 = plt.subplots()
plot_df_top(df, fig1, ax1, color=cmap(norm(H[0])), label=rf'$H = {H[0]*1e3:.2f} \, mm$')

abs_path = "S:\\masterproject\\images\\height_measurements\\11042024\\set2\\"
file = os.path.join(abs_path, 'data.pkl')
df = pd.read_pickle(file)
plot_df_top(df, fig1, ax1, color=cmap(norm(H[1])), label=rf'$H = {H[1]*1e3:.2f} \, mm$')

# abs_path = "S:\\masterproject\\images\\height_measurements\\11042024\\set3\\"
# file = os.path.join(abs_path, 'data.pkl')
# df = pd.read_pickle(file)
# plot_df_top(df, fig1, ax1, color='green')

abs_path = "S:\\masterproject\\images\\height_measurements\\11042024\\set4\\"
file = os.path.join(abs_path, 'data.pkl')
df = pd.read_pickle(file)
plot_df_top(df, fig1, ax1, color=cmap(norm(H[2])), label=rf'$H = {H[2]*1e3:.2f} \, mm$')

abs_path = "S:\\masterproject\\images\\height_measurements\\11042024\\set5\\"
file = os.path.join(abs_path, 'data.pkl')
df = pd.read_pickle(file)
plot_df_top(df, fig1, ax1, color=cmap(norm(H[2])), label=rf'$H = {H[2]*1e3:.2f} \, mm$')

abs_path = "S:\\masterproject\\images\\height_measurements\\11042024\\set6\\"
file = os.path.join(abs_path, 'data.pkl')
df = pd.read_pickle(file)
plot_df_top(df, fig1, ax1, color=cmap(norm(H[3])), label=rf'$H = {H[3]*1e3:.2f} \, mm$')

abs_path = "S:\\masterproject\\images\\dodecane_17012025\\set2\\"
file = os.path.join(abs_path, 'data.pkl')
df = pd.read_pickle(file)
plot_df_top(df, fig1, ax1, color=cmap(norm(H[4])), label=rf'$H = {H[4]*1e3:.2f} \, mm$')


