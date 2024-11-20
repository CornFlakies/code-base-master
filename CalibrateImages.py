# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 21:30:22 2024

@author: Coen Arents
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
from skimage.measure import find_contours

def normalize_img(img):
    return (img - img.min())/img.max()

def connectedEdges(c):
    '''
    Check if the edges of the contour are connected
    '''
    if (np.linalg.norm(c[0] - c[-1]) < 5):
        return True
    
def find_center(c):
    cx = np.mean(c[:, 1])
    cy = np.mean(c[:, 0])
    return cx, cy
    
path125micron = "C:\\Users\\Coen Arents\\Desktop\\measurements\\31-10-2024_set2\\calibration_top_view_0125mm.tif"
path500micron = "C:\\Users\\Coen Arents\\Desktop\\measurements\\31-10-2024_set2\\calibration_top_view_0500mm.tif"
path125micron = "C:\\Users\\Coen Arents\\Desktop\\measurements\\31-10-2024_set2\\calibration_side_view_0125mm.tif"
path500micron = "C:\\Users\\Coen Arents\\Desktop\\measurements\\31-10-2024_set2\\calibration_side_view_0500mm.tif"

img125 = normalize_img(sk.io.imread(path125micron))[300:800, 300:800]
img500 = normalize_img(sk.io.imread(path500micron))

#%% Alpha 125 micron spacing
plt.close('all')
contours = find_contours(img125)
centers = []

plt.imshow(img125, cmap='gray')
for ii, c in enumerate(contours):
    if (len(c) < 50):
        continue
    plt.plot(c[:, 1], c[:, 0], color='red')
    
    cx, cy = find_center(c)
    plt.plot(cx, cy, '.', color='blue')
    centers.append(np.array([cx, cy]))

dist = []
for ii in range(1, len(centers)):
    dist.append(np.linalg.norm(centers[ii-1] - centers[ii]))

mask =[]
for ii, d in enumerate(dist):
    if (d > (dist[0] + 10)):
        mask.append(True)
    else:
        mask.append(False)
    
        
dist = np.ma.array(dist, mask=mask)
    
alpha125 = (0.125 / dist).mean() #mm px**-1
stdalpha125 = (0.125 / dist).std() #mm px**-1
print(str(alpha125 * 1e3) + ' pm ' + str(stdalpha125 * 1e3) + ' micron per pixel')

#%% Alpha 500 micron spacing
plt.close('all')
contours = find_contours(img500)
centers = []

plt.imshow(img500, cmap='gray')
avg_length = []
for c in contours:
    avg_length.append(len(c))

avg_length = np.mean(avg_length)
print(avg_length)

for ii, c in enumerate(contours):
    if (len(c) < avg_length + 20):
        continue
    plt.plot(c[:, 1], c[:, 0], color='red')
    cx, cy = find_center(c)
    # print(cx, cy)
    plt.plot(cx, cy, '.', color='blue')
    centers.append(np.array([cx, cy]))

dist = []
for ii in range(1, len(centers)):
    dist.append(np.linalg.norm(centers[ii-1] - centers[ii]))
    # for jj in range(ii, len(centers)):
    #     dist.append(np.linalg.norm(centers[ii] - centers[jj]))

mask =[]
for ii, d in enumerate(dist):
    if (d > (dist[0] + 10)):
        mask.append(True)
    else:
        mask.append(False)
        
dist = np.ma.array(dist, mask=mask)
    
alpha500 = (0.500 / dist).mean() #mm px**-1
stdalpha500 = (0.500 / dist).std()
print(str(alpha500 * 1e3) + ' pm ' + str(stdalpha500 * 1e3) + ' micron per pixel')



