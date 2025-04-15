# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 18:30:27 2025

@author: coena
"""

import os
import re
import yaml
import numpy as np
import pandas as pd
import skimage as sk
from tqdm import tqdm
import HelperFunctions as hp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
from ComputeLensDynamics import ComputeLensDynamics

abs_path_1 = "D:\\masterproject\\images\\dodecane_17012025\\set2"
file = os.path.join(abs_path_1, 'data.pkl')
df1 = pd.read_pickle(file)

abs_path_2 = "D:\\masterproject\\images\\dodecane_18032025\\set1"
file = os.path.join(abs_path_2, 'data.pkl')
df2 = pd.read_pickle(file)

abs_path_3 = "D:\\masterproject\\images\\dodecane_20032025\\set1"
file = os.path.join(abs_path_3, 'data.pkl')
df3 = pd.read_pickle(file)

abs_path_4 = "D:\\masterproject\\images\\dodecane_26032025\\set2"
file = os.path.join(abs_path_4, 'data.pkl')
df4 = pd.read_pickle(file)

abs_path_5 = "D:\\masterproject\\images\\mineral_oil_27032025_94mpas\\set2_zoom"
file = os.path.join(abs_path_5, 'data.pkl')
df5 = pd.read_pickle(file)

abs_path_6 = "D:\\masterproject\\images\\mineral_oil_27032025_94mpas\\set2_zoom_highest_fps"
file = os.path.join(abs_path_6, 'data.pkl')
df6 = pd.read_pickle(file)

abs_path_7 = "D:\\masterproject\\images\\mineral_oil_27032025_94mpas\\set1_no_zoom\\"
file = os.path.join(abs_path_7, 'data.pkl')
df7 = pd.read_pickle(file)

plt.close('all')

R = 1e-3 # m

# DODECANE
eta = 1.36e-3 # Pa s
gamma =  25e-3 # N / m
rho = 750 # kg / m^3
tau_v = eta * R / gamma # Viscous length scale of dodecane
Oh = eta / np.sqrt(rho * gamma * R) # Ohnesorge number dodecane
l_c = R * Oh # Crossover length dodecame
t_c = tau_v * Oh # Crossover time dodecane

# WATER
eta0 = 1e-3 # Pa s
gamma0 = 72e-3 # N / m
rho0 = 998 # kg / m^3

# MINERAL OIL 94 mPa
eta1 = 94e-3
gamma1 = 25e-3
rho1 = 818 # kg / m^3
tau_v1 = eta1 * R / gamma1 # Viscous length scale of dodecane
Oh1 = eta1 / np.sqrt(rho1 * gamma1 * R) # Ohnesorge number dodecane
l_c1 = R * Oh1 # Crossover length dodecame
t_c1 = tau_v1 * Oh1 # Crossover time dodecane

fig1, ax1 = plt.subplots()


r_and_t = []

# Loop measurement realizations
for ii in df1.index:    
    # Load all the data from the dataframe
    r_max_top = df1.loc[ii, 'R_max_top']
    frames_top = df1.loc[ii, 'frames_top']
    R = df1.loc[ii, 'drop_radii']
    fps = df1.loc[ii, 'fps']
    alpha_top = df1.loc[ii, 'alpha_top']
    
    r_and_t.append([r_max_top, frames_top / fps, 'red'])
    r = np.linalg.norm(r_max_top - r_max_top[0], axis=1)
    t = (frames_top - frames_top[0]) / fps
    ax1.loglog(t[1:] / t_c, r[1:] / l_c, 'o', color='green')
    
for ii in df3.index:
    # Load all the data from the dataframe
    r_max_top = df3.loc[ii, 'R_max_top']
    frames_top = df3.loc[ii, 'frames_top']
    R = df3.loc[ii, 'drop_radii']
    fps = df3.loc[ii, 'fps']
    alpha_top = df3.loc[ii, 'alpha_top']
    if (ii == 2):
        continue
    r_and_t.append([r_max_top, frames_top / fps, 'blue'])
    r = np.linalg.norm(r_max_top - r_max_top[0], axis=1)
    t = (frames_top - frames_top[0]) / fps
    ax1.loglog(t[1:] / t_c, r[1:] / l_c, 'o', color='green')
    
for ii in df4.index:
    # Load all the data from the dataframe
    r_max_top = df4.loc[ii, 'R_max_top']
    frames_top = df4.loc[ii, 'frames_top']
    R = df4.loc[ii, 'drop_radii']
    fps = df4.loc[ii, 'fps']
    alpha_top = df4.loc[ii, 'alpha_top']
    r_and_t.append([r_max_top, frames_top / fps, 'green'])
    r = np.linalg.norm(r_max_top - r_max_top[0], axis=1)
    t = (frames_top - frames_top[0]) / fps
    ax1.loglog(t[1:] / t_c, r[1:] / l_c, 'o', color='green')

# Loop measurement realizations
for ii in df5.index:    
    # Load all the data from the dataframe
    r_max_top = df5.loc[ii, 'R_max_top']
    frames_top = df5.loc[ii, 'frames_top']
    R = df5.loc[ii, 'drop_radii']
    fps = df5.loc[ii, 'fps']
    alpha_top = df1.loc[ii, 'alpha_top']
    if (ii == 2):
        continue
    r_and_t.append([r_max_top, (frames_top / fps), 'blue'])
    r = np.linalg.norm(r_max_top - r_max_top[0], axis=1)
    t = (frames_top - frames_top[0]) / fps
    ax1.loglog(t[1:] / t_c1, r[1:] / l_c1, 'o', color='blue')

for ii in df7.index:
    # Load all the data from the dataframe
    r_max_top = df7.loc[ii, 'R_max_top']
    frames_top = df7.loc[ii, 'frames_top']
    R = df7.loc[ii, 'drop_radii']
    fps = df7.loc[ii, 'fps']
    alpha_top = df7.loc[ii, 'alpha_top']
    if (ii == 1):
        continue
    r_and_t.append([r_max_top, (frames_top / fps), 'blue'])
    r = np.linalg.norm(r_max_top - r_max_top[0], axis=1)
    t = (frames_top - frames_top[0]) / fps + 5e-6
    ax1.loglog(t[1:] / t_c1, r[1:] / l_c1, 'o', color='blue')

    
    
    
    