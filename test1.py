# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:50:06 2025

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

path_burton = "D:\\masterproject\\scrubbed_data\\burton\\dodecane_01.csv"
data_burton2 = pd.read_csv(path_burton).to_numpy() # centimeters and second
data_burton2[:, 0] *= 1e6
data_burton2[:, 1] *= 1e4
path_burton = "D:\\masterproject\\scrubbed_data\\burton\\dodecane_02.csv"
data_burton3 = pd.read_csv(path_burton).to_numpy() # centimeters and second
data_burton3[:, 0] *= 1e6
data_burton3[:, 1] *= 1e4

t0 = np.linspace(-200, 0, 5)
# t0 = [-350.0]

norm = mcolors.LogNorm(vmin=0, vmax=len(t0))
norm = ['blue', 'red', 'green', 'purple', 'orange']
cmap = plt.cm.Reds

plt.close('all')
fig, ax = plt.subplots()

for ii, t in enumerate(t0):
    ax.loglog(data_burton2[:, 0] + t , data_burton2[:, 1], 'd', color=norm[ii], label=rf'Burton with $t* = {t:0.2f}$')
    ax.loglog(data_burton3[:, 0] + t, data_burton3[:, 1], 'd', color=norm[ii])

ax.legend()


