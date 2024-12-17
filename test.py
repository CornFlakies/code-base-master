# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:10:02 2024

@author: coena
"""
import os
import time
import numpy as np
from tqdm import tqdm
import DetectSideView as dps
import DetectTopView as dpt
import HelperFunctions as hp
import matplotlib.pyplot as plt
from ComputeLensDynamics import ComputeLensDynamics

plt.close('all')

path = "C://Users//coena//OneDrive - University of Twente//universiteit//master//master_project//WORK//"
path = "C:\\Users\\coena\\WORK\\master\\images\\side_top_view_video1_20240919_170427"
# path = "C:\\Users\\coena\\WORK\\master\\images\\24-09-2024_mineral_oil\\coalescence_95mPas_video6_20240909_111139"
# path2 = "C:\\Users\\coena\\OneDrive - University of Twente\\universiteit\\master\\master_project\\WORK\\images\\test_videos\\coalescence_95mPas_video6_20240909_151353"
path = "C:\\Users\\coena\\WORK\\master\\images\\31-10-2024_dodecane\\31-10-2024_set2\\top_nd_side_coalescence_9_nice\\top_view"
path = "D:\\masterproject\\images\\dodecane_coalescence\\31-10-2024_set2\\top_nd_side_coalescence_10_nice\\top_view"
# path = "C:\\Users\\Coen Arents\\Desktop\\measurements\\31-10-2024_set2\\top_nd_side_coalescence_9_nice\\top_view"
# path = "C:\\Users\\Coen Arents\\Desktop\\measurements\\31-10-2024_set2\\top_nd_side_coalescence_10_nice\\top_view"
# path = "C:\\Users\\coena\\WORK\\master\\images\\31-10-2024_dodecane\\31-10-2024_set2\\top_nd_side_coalescence_9_nice\\side_view"

# Determine window of interest
XMIN = 0
XMAX = None
YMIN = 0
YMAX = None

# Get framestart
framestart=80
image_paths, _ = hp.load_files(path)
image = hp.load_from_stack(image_paths[0], framestart)[YMIN:YMAX, XMIN:XMAX]


cpd = ComputeLensDynamics(path, XMIN, XMAX, YMIN, YMAX, framestart=('stack 1', framestart), view='top')
r_max = cpd.get_R()

#%% Plot maxima
def power_law(A, x, p):
    return A * (x/x[0])**(p)

plt.close('all')

alpha = 3.471442521590393 #micrometer

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

#%% Plot the maxima
plt.close('all')

def power_law(A, x, p):
    return A * (x/x[0])**(p)

fps = 20E3

r_max_vector = np.zeros((len(r_max), 2))
x_experimental = np.arange(0, r_max_vector.shape[0]) * 5 / fps * 1E3
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
plt.title(r'Time evolution of $h(t)$, mineral oil 95 $mPa\cdot s,\, Oh\approx 0.6$')
plt.loglog(np.abs(r_max_vector[:, 1] - r_max_vector[0, 1]), x_experimental, '.-', label='experimental data')
plt.loglog(x_analytical, power_law(1e0, x_analytical, 1), '--', color='black', label=r'$\sim t$')
plt.legend()
plt.xlabel('t (ms)')
plt.ylabel('h(t) - h(0) (px)')
plt.grid()



    
