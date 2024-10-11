# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:10:02 2024

@author: coena
"""
import os
import time
import numpy as np
from tqdm import tqdm
import DetectDroplets as dp
import HelperFunctions as hp
import matplotlib.pyplot as plt
from ComputeLensDynamics import ComputeLensDynamics

plt.close('all')

path = "C://Users//coena//OneDrive - University of Twente//universiteit//master//master_project//WORK//"
path = "C:\\Users\\coena\\WORK\\master\\images\\side_top_view_video1_20240919_170427"
path = "C:\\Users\\coena\\WORK\\master\\images\\coalescence_95mPas_video6_20240909_111139"
path2 = "C:\\Users\\coena\\OneDrive - University of Twente\\universiteit\\master\\master_project\\WORK\\images\\test_videos\\coalescence_95mPas_video6_20240909_151353"

# Determine window of interest
XMIN = 580
XMAX = 640
YMIN = 550
YMAX = 640

image_paths, _ = hp.load_files(path)
image = hp.load_from_stack(image_paths[0], 50)[YMIN:YMAX+1, XMIN:XMAX+1]

t_init = time.time()
edges, canny_edges = dp.detect_edges(image)
t_final1 = time.time() - t_init
print('Edge detection run took: ' + str(t_final1*1e3) + ' ms')

t_init = time.time()
x_max, y_max, spline = dp.find_edge_extrema(edges)
t_final2 = time.time() - t_init
print('Maximum finding run took: ' + str(t_final2*1e3) + ' ms')

print('Total time of: ' + str((t_final1 + t_final2) * 1e3) + 'ms')

plt.figure()
plt.imshow(image, cmap='gray')
for coord in canny_edges:
    plt.plot(coord[0], coord[1], '.', color='lime')
for coord in edges:
    plt.plot(coord[0], coord[1], '.', color='red')
    
# Plot spline
xsp = np.linspace(edges[0, 0], edges[-1, 0], int(1E6))
plt.plot(xsp, spline(xsp), '-', color='red')

# Plot maximum
plt.plot(x_max, y_max, marker='x', color='yellow')
    
# Plot the derivative
# plt.figure()
# plt.plot(xsp, spline.derivative()(xsp), '-', color='blue')

# cpd = ComputeLensDynamics(path, XMIN, XMAX, YMIN, YMAX)
# r_max = cpd.get_R



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



    
