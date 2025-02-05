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

# If the config files do not exist, create new ones.
isLive = True
if isLive:
    fps = 1e5
    iim.create_config_from_suite(abs_path, fps)
    


#%% Plot powerlaw
def power_law(A, x, p):
    return A * (x/x[0])**(p)

plt.close('all')

maxima = r_max[0]
lower_max = np.asarray(maxima[0])
higher_max = np.asarray(maxima[1])
lower_max_init = lower_max
higher_max_init = higher_max

maxima = r_max[-1]
lower_max = np.asarray(maxima[0])
higher_max = np.asarray(maxima[1])
higher_max_final  = np.asarray(maxima[0])
lower_max_final = np.asarray(maxima[1])
p1_max = []
p2_max = []

plt.figure()
plt.imshow(image)
for ii in range(len(r_max)):
    maxima = r_max[ii]
    lower_max = np.asarray(maxima[0])
    higher_max = np.asarray(maxima[1])

    p1_max.append(np.linalg.norm((higher_max - higher_max_init) * alpha)) # / higher_max_final))
    p2_max.append(np.linalg.norm((lower_max - lower_max_init) * alpha)) # / higher_max_final))
    
    # p1_max.append(np.abs(lower_max[1] - lower_max_init[1]))
    # p2_max.append(np.abs(higher_max[1] - higher_max_init[1]))
    plt.xlim([0, 300])
    plt.plot(higher_max[0], higher_max[1], '.', color='blue')
    plt.plot(higher_max[0], lower_max[1], '.', color='red')
    
x_ana = np.arange(0, len(r_max)) / 100000 * 1e6

plt.figure()
plt.loglog(x_ana[1:], power_law(p1_max[7], x_ana[1:], 1/2), '--', color='grey', label=r'$r \sim t^{1/2}$')
plt.loglog(x_ana[1:], power_law(p1_max[1], x_ana[1:], 1), '--', color='black', label=r'$r \sim t$')
plt.loglog(x_ana, p1_max, 'o', color='blue', label='data')
plt.title('Neck radius as a function of time')
plt.legend()
plt.ylim([10**(0.5), 1e3])
plt.xlim([1e1, 1e4])
plt.xlabel(r'$t\, [\mu s]$')
plt.ylabel(r'$h_0(t)\, [\mu m]$')
plt.figure()
plt.loglog(x_ana[1:], power_law(p2_max[7], x_ana[1:], 1/2), '--', color='grey', label=r'$r \sim t^{1/2}$')
plt.loglog(x_ana[1:], power_law(p2_max[1], x_ana[1:], 1), '--', color='black', label=r'$r \sim t$')
plt.loglog(x_ana, p2_max, 'o', color='blue', label='data')
plt.title('Neck radius as a function of time')
plt.legend()
plt.ylim([10**(0.5), 1e3])
plt.xlim([1e1, 1e4])
plt.xlabel(r'$t\, [\mu s]$')
plt.ylabel(r'$h_0(t)\, [\mu m]$')