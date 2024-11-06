# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:05:45 2024

@author: coena
"""

import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
from skimage.measure import find_contours


def contour_edges(image):
    '''
    Routine used to find the contours of an image
    '''
    # Gaussian blurring
    image = sk.filters.gaussian(image, sigma=1)
    
    # Normalize to 1
    image = (image - image.min()) / image.max()
    
    # Get contours 
    contours = find_contours(image)
    
    return contours
    

def canny_edges(image):
    '''
    Routine to compute the Canny edges of an image.
    '''
    # Gaussian blurring
    image = sk.filters.gaussian(image, sigma=1)

    # Normalize to 1
    image = ((image - image.min()) / image.max())
    
    # Canny params.
    # defaults are: t_min = 0.1 * img_max ; t_max = 0.2 * img_max
    t_min = None
    t_max = None
    
    # Apply Canny edge detector
    edges = sk.feature.canny(image, 
                             sigma=0,
                             low_threshold=t_min,
                             high_threshold=t_max)

    return edges

def detect_edges(image):
    '''
    Takes a single (8-bit, grayscale) image and detects the edge(s) of the droplets.
    To get supbixel accuracy, a Gaussian interpolation is done along the edge pixels
    detected by a Canny edge detector.
    '''
    # Get contours
    contours = find_contours(image)
    
    # Midline image
    midline = image.shape[1] // 2
    
    print(midline)
    
    return contours

def is_connected(image):
    '''
    Find the first frame where the droplets connect, such that a starting frame
    can be defined, from which to base the initial bridge height off.
    '''
    # Get Canny edges
    edges = canny_edges(image)
    
    # Flatten edge array
    edge_flat = np.mean(edges, axis=0)
    edge_flat = np.where(edge_flat > 0, 1, 0)
    
    # Ignore leading and trailing zeros by slicing the array up to the last 1
    first_one_index = np.min(np.where(edge_flat == 1)[0]) if 1 in edge_flat else -1
    last_one_index = np.max(np.where(edge_flat == 1)[0]) if 1 in edge_flat else -1
    edge_flat = edge_flat[first_one_index:last_one_index + 1] if first_one_index != -1 else edge_flat
    
    # Find the amount of broken sequences in the flattened array
    start_of_breaks = (edge_flat[:-1] == 1) & (edge_flat[1:] == 0)
    num_breaks = np.sum(start_of_breaks)
    
    if (num_breaks == 0):
        return True
    else:
        return False

def find_edge_extrema(contours):
    '''
    Finds the maximum location based on the subpixel detected edge, a parameterization
    scheme is used to prepare the x- and y-data for cubic spline 
    interpolation (such that the data scales monotonically). The resultant spline \
    is then used to get the maximum location of the droplet bridge
    '''
    x_max = []
    y_max = []
    for c in contours:
        # Separate x- and y-coordinates for readability
        x = c[:, 1]
        y = c[:, 0]    
        
        plt.figure()
        plt.plot(x, y)
    
        # # Fit spline
        # spline = UnivariateSpline(x, y)
        # roots = spline.deriv().roots()
    
        # x_max.append(roots)
        # y_max.append(spline(roots))
    
    return x_max, y_max
