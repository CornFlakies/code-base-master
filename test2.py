# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 14:07:13 2025

@author: Coen Arents
"""

import os
import numpy as np
import skimage as sk
import matplotlib.pyplot as plt

def normalize_img(img):
    return (img - img.min())

path = "S:\\masterproject\\images\\height_measurements\\11042024\\set4\\"
file = "water_height_calibration_250micron_side.tif"
# file = "water_height_C001H001S0001.tif"
file_path = os.path.join(path, file)

img = sk.io.imread(file_path)
img = normalize_img(img[:, 400:-20, 1])

# Load in top view contours, to show for the plots


# Load in side view contours, to show for the plots