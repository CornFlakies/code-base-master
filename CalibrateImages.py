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
        # plt.imshow(self.img, cmap='gray')
        
        all_lengths = []
        for c in contours:
            all_lengths.append(len(c))
        max_length = np.mean(all_lengths)
        
        for ii, c in enumerate(contours):
            
            if (len(c) < (max_length - (max_length/20))):
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
    
        return (self.spacing / dist).mean() #m px**-1

import os

abs_path = "S:\\masterproject\\images\\height_measurements\\11042024\\set6\\water_height\\"
file =  "water_height_calib_side_500micron_20250508_125957000001.tif"
calib_path = os.path.join(abs_path, file)
calib_img = sk.io.imread(calib_path)
plt.close('all')

# # Load the files and directories from path
# contents = os.listdir(abs_path)

# Get images for calibration, define alpha of both top and side view
calib_paths, calib_files = hp.load_files(abs_path, header="tif")
ci = CalibrateImages()
alpha = ci.run(calib_path, file)

# # # Multiple calibration images can be provided
# # alpha_top = []
# # alpha_side = []
# # for path, file in zip(calib_paths, calib_files):
# #     view = re.search("top|side", file).group()
# #     dist = ci.run(path, file)
# #     if (view == "top"):
# #         alpha_top.append(dist)
# #     elif (view == "side"):
# #         alpha_side.append(dist)
        
# print(alpha)