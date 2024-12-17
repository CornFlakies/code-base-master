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
    '''
    Normalize img to unity
    '''
    return (img - img.min())/img.max()

class CalibrateImages():
    def __init__(self, img, spacing, type='dots'):
        self.img = img
        self.spacing = spacing
        self.type = type
        
        
    def run(self):
        if self.type == 'dots':
            return self.compute_dot_distances(self.img, self.spacing)
    
    def compute_dot_distances(self):
        def connectedEdges(c):
            '''
            Check if the edges of the contour are connected
            '''
            if (np.linalg.norm(c[0] - c[-1]) < 5):
                return True
        
        def find_center(c):
            '''
            find center of each circle
            '''
            cx = np.mean(c[:, 1])
            cy = np.mean(c[:, 0])
            return cx, cy
        '''
        Remove cut-off contours
        '''
        contours = find_contours(self.img)
        centers = []
        
        avg_length = []
        for c in contours:
            avg_length.append(len(c))
        avg_length = np.mean(avg_length)
    
        for ii, c in enumerate(contours):
            if (len(c) < avg_length + 2):
                continue
            plt.plot(c[:, 1], c[:, 0], color='red')
            cx, cy = find_center(c)
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
    
        return (self.spacing / dist).mean() #mm px**-1
    

# path125micron = "D:\\masterproject\\images\\dodecane_31102024\set2\\calibration_top_view_0125mm.tif"
# path500micron = "D:\\masterproject\\images\\dodecane_31102024\\set2\\calibration_top_view_0500mm.tif"

# img125 = normalize_img(sk.io.imread(path125micron))
# dist125 = compute_dot_distances(img125)
# alpha125 = (0.125 / dist125).mean() #mm px**-1
# stdalpha125 = (0.125 / dist125).std() #mm px**-1
# print(str(alpha125 * 1e3) + ' pm ' + str(stdalpha125 * 1e3) + ' micron per pixel')

# img500 = normalize_img(sk.io.imread(path500micron))
# dist500 = compute_dot_distances(img500)
# alpha500 = (0.500 / dist500).mean() #mm px**-1
# stdalpha500 = (0.125 / dist500).std() #mm px**-1
# print(str(alpha500 * 1e3) + ' pm ' + str(stdalpha500 * 1e3) + 'micron per pixel')

# path125micron = "D:\\masterproject\\images\\dodecane_31102024\set2\\calibration_side_view_0125mm.tif"
# path500micron = "D:\\masterproject\\images\\dodecane_31102024\\set2\\calibration_side_view_0500mm.tif"

# img125 = normalize_img(sk.io.imread(path125micron))
# dist125 = compute_dot_distances(img125)
# alpha125 = (0.125 / dist125).mean() #mm px**-1
# stdalpha125 = (0.125 / dist125).std() #mm px**-1
# print(str(alpha125 * 1e3) + ' pm ' + str(stdalpha125 * 1e3) + ' micron per pixel')

# img500 = normalize_img(sk.io.imread(path500micron))
# dist500 = compute_dot_distances(img500)
# alpha500 = (0.500 / dist500).mean() #mm px**-1
# stdalpha500 = (0.125 / dist500).std() #mm px**-1
# print(str(alpha500 * 1e3) + ' pm ' + str(stdalpha500 * 1e3) + 'micron per pixel')


