# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 10:23:06 2025

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

def refine_t0(xdata, ydata, start, end, p=1/2):
    power_law = lambda x, A: A * x**p
    popt, _ = curve_fit(power_law, xdata[start:end], ydata[start:end])
    A = popt[0]
    print(A)
    power_law = lambda x, x0: A * (x - x0)**p
    popt, _ = curve_fit(power_law, xdata, ydata)
    
    return None


abs_path_4 = "D:\\masterproject\\images\\dodecane_26032025\\set2"
file = os.path.join(abs_path_4, 'data.pkl')
df4 = pd.read_pickle(file)

r_and_t = []

for ii in df4.index:
    # Load all the data from the dataframe
    r_max_top = df4.loc[ii, 'R_max_top']
    frames_top = df4.loc[ii, 'frames_top']
    R = df4.loc[ii, 'drop_radii']
    fps = df4.loc[ii, 'fps']
    alpha_top = df4.loc[ii, 'alpha_top']
    if (ii == 2):
        continue
    r_and_t.append([r_max_top, frames_top / fps, 'green'])

r, t, c = r_and_t[0]
r = r - r[0]
t = t - t[0]
r = r[1:, 1]
t = t[1:]

plt.close('all')
plt.figure()
plt.loglog(t, r, 'o')

refine_t0(t, r, -10, None, p=1/2)

# refine_t0(t, r, )