# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:48:22 2024

@author: coena
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import corner_peaks
from skimage.feature import corner_subpix
import scipy.interpolate

def canny_edges(image, T_min, T_max, kernel):
    '''
    Routine to compute the Canny edges of the image.
    '''
    # Remove noise with a gaussian blur filter
    blur = cv.GaussianBlur(image, kernel, 0)
    
    # Apply Canny edge detector
    edges = cv.Canny(blur, T_min, T_max)
    
    return edges    

def detect_edges(image, T_min=50, T_max=170, kernel=(3,3)):
    '''
    Takes a single (8-bit, grayscale) image and detects the edge(s) of the droplets 
    This is done using subpix accuracy, as such the algorithm spits out a vector
    of (x,y) coordinates of the edges.
    '''
    
    # Transpose image so that the image is oriented correctly for edge detection
    image = image.T
    
    # Get Canny edges
    edges = canny_edges(image, T_min, T_max, kernel)
    
    # Find initial edge points
    coords = np.column_stack(np.where(edges > 0))
    
    # Filter out the edge coordinates at the boundary of the image, otherwise
    # There will be a mismatch betwen the windowing and the edge coordinates
    padding = 7 #px
    coords = coords[
        (coords[:, 0] >= padding) & (coords[:, 0] < image.shape[0] - padding) &
        (coords[:, 1] >= padding) & (coords[:, 1] < image.shape[1] - padding)
    ]
    
    # Apply subpixel refinement using Devernay approximation algorithm
    coords_subpix = corner_subpix(image, coords, window_size=13)
    
    return coords_subpix

def find_edge_extrema(coords_edges):
    '''
    Finds the maximum location based on the subpixel detected edge, a parameterization
    scheme is used to prepare the x- and y-data for interpolation (such that the data
    scales monotonically). The resultant spline is then used to get the maximum 
    location of the droplet bridge
    '''
    # sort_idx = np.argsort(coords_edges[]xis=0)
    # x = coords_edges[sort_idx, 0]
    # y = coords_edges[sort_idx, 1]
    # spline = scipy.interpolate.CubicSpline(x, y)
    
    x = coords_edges[:, 0]
    y = coords_edges[:, 1]
    t = np.linspace(0, 1, x.size)
    r = np.vstack((x.reshape((1, x.size)), y.reshape((1, y.size))))
    spline = scipy.interpolate.interp1d(t, r, kind='cubic')
    
    # you want to interpolate x and y
    # it MUST be within 0 and 1, since you defined
    # the spline between path_t=0 and path_t=1
    t = np.linspace(np.min(t), np.max(t), 100)
    
    # interpolating along t
    # r[0,:] -> interpolated x coordinates
    # r[1,:] -> interpolated y coordinates
    
    # First iteration, get rough estimate of the maximum
    r = spline(t)
    t_max = t[np.argmax(r[1, :])]
    
    # Second iteration, refine grid and get better estimate of the maximum.
    buffer = 1E-2
    t = np.linspace(t_max - buffer, t_max + buffer, int(1E6))
    r = spline(t)
    x_max = r[0, np.argmax(r[1, :])]
    y_max = np.max(r[1, :])
    
    # print(x_max)
    # print(y_max)
    # plt.figure()
    # plt.plot(r[0, :], r[1, :], '.-')
    # plt.plot(x, y, '-')
    
    return x_max, y_max
    
def is_connected(image, T_min=50, T_max=170, kernel=(3,3)):
    '''
    find the first frame where the droplets connect, such that a starting frame
    can be defined, from which to base the initial bridge height off.
    '''
    # Get canny edges
    edges = canny_edges(image, T_min, T_max, kernel)
    
    # Find the two edges, if they exist
    gap = 0
    for col in range(edges.shape[1]):
        if not any(edges[:, col] > 0):
            gap += 1
    
    # Check if the gap has closed, then return True
    if (gap == 0):
        return True 
    else:
        return False
    
        