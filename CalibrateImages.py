# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 21:30:22 2024

@author: Coen Arents
"""

import re
import numpy as np
import skimage as sk
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
import HelperFunctions as hp
import matplotlib.pyplot as plt
from skimage.measure import find_contours

def normalize_img(img):
    '''
    Normalize img to unity
    '''
    return (img - img.min()) / img.max()

class CalibrateImages():
        
    def get_spacing_from_file(self, file):
        # either top or side calib file
        if (hp.file_contains_substring(file, "top")):
            calib = "top"
        elif (hp.file_contains_substring(file, "side")):
            calib = "side"
        else:
            raise Exception ('calibration file name does not specify if it is top or side view')
        
        # Get the dot spacing from the calibration file
        spacing_str = re.search('[0-9]+mm|[0-9]+micron', file).group()
        
        # Convert unit_str to number
        unit_str = re.search('[a-z]+', spacing_str).group()
        if unit_str == "micron":
            unit = 1e-6 
        elif unit_str == "mm":
            unit = 1e-3
            
        spacing = float(re.search('[0-9]+', spacing_str).group()) * unit
        return spacing
    
    def run(self, path, file):
        self.spacing = self.get_spacing_from_file(file)
        self.img = sk.io.imread(path)
        return self.compute_dot_distances()
    
    def compute_dot_distances(self):
        def isConnected(c):
            '''
            Check if the edges of the contour are connected
            '''
            return None
        
        def find_center(c):
            '''
            find center of each circle
            '''
            cx = np.mean(c[:, 1])
            cy = np.mean(c[:, 0])
            return cx, cy

        contours = find_contours((1 - normalize_img(self.img)))
        centers = []
        
        # plt.figure()
        # plt.imshow(self.img)
        
        all_lengths = []
        for c in contours:
            all_lengths.append(len(c))
        max_length = np.max(all_lengths)
        
        for ii, c in enumerate(contours):
            if (len(c) < (max_length - (max_length/10))):
                continue
            # plt.plot(c[:, 1], c[:, 0], color='red')
            cx, cy = find_center(c)
            # plt.plot(cx, cy, '.', color='blue')
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

