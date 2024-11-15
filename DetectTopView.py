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

def connectedEdges(c):
    '''
    Check if the edges of the contour are connected
    '''
    if (np.linalg.norm(c[0] - c[-1]) < 10):
        return True

def sortCont(c):
    '''
    Sort contour from lowest to highest x-values 
    '''
    c[:, 0] = c[np.argsort(c[:, 1]), 0]
    c[:, 1] = c[np.argsort(c[:, 1]), 1]
    return c

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
        plt.figure()
        plt.imshow(image)
        return True
    else:
        return False

def find_edge_extrema(image, contours):
    '''
    Finds the maximum location based on the subpixel detected edge, a parameterization
    scheme is used to prepare the x- and y-data for cubic spline 
    interpolation (such that the data scales monotonically). The resultant spline \
    is then used to get the maximum location of the droplet bridge
    '''
    
    # Get the padding of the positively curved line around the maximum
    def getPadding(c, idx_max):
        pad = 0
        ii = idx_max
        # print(c[:, 1])
        # Find the strictly decreasing points 
        while ii < (len(c[:, 1]) - 1):
            if (c[ii, 1]) < (c[ii + 1, 1]):
                pad += 1
                ii += 1
            else:
                break
        pad_f = pad
        pad = 0
        ii = idx_max
        # Find the strictly decreasing points
        while ii > 0:
            if (c[ii, 1] > c[ii - 1, 1]):
                pad += 1
                ii -= 1
            else:
                break
        pad_b = pad
        return pad_f, pad_b
    
    # Get minimum of negatively curved line
    def getMax(c, idx_max):
        pad_f, pad_b = getPadding(c, idx_max)
        
        x = c[(idx_max - pad_b):(idx_max + pad_f), 1]
        y = c[(idx_max - pad_b):(idx_max + pad_f), 0]
        x_ana = np.linspace(x[0], x[-1], 100)
        
        # plt.figure()
        # plt.plot(x, y, '.-')

        spline = CubicSpline(x, y)
        spline_xmax = spline.derivative().roots()
        spline_ymax = spline(spline_xmax)
        
        if len(spline_xmax) > 1:
            closest_max = np.abs(spline_ymax - c[idx_max, 0])
            spline_ymax = spline_ymax[np.argmin(closest_max)]
            spline_xmax = spline_xmax[np.argmin(closest_max)]
            
        # plt.figure()
        # plt.plot(x_ana, spline(x_ana))
        # plt.plot(spline_xmax, spline_ymax, 'o', color='red')
            
        return spline_xmax, spline_ymax
    
    # Log the maxima of the two contours
    c_max = []
    
    # Plot contours for debugging pruposes
    # plt.figure()
    # plt.axhline(image.shape[0] // 2)
    # for c in contours:
    #     plt.plot(c[:, 1], c[:, 0], '.-')
    
    # Find midline, such that contours above and below can be identified
    midline = image.shape[0] // 2
    for ii, c in enumerate(contours):
        ct = c[:, 0] < midline
        cb = c[:, 0] > midline
        # Remove the circular contour crossing the midline
        if (any(cb) and any(ct)) or (connectedEdges(c)):
            continue
        # If contour above midline
        if all(ct) and not all(cb):
            c = sortCont(c)
            idx_max = np.argmax(c[:, 0])
            # print(c[idx_max, 0])
            cmax = getMax(c, idx_max)
            c_max.append(cmax)
            print('min')
        # If contour below midline
        elif all(cb) and not all(ct):
            c = sortCont(c)
            idx_max = np.argmin(c[:, 0])
            # print(c[idx_max, 0])
            cmin = getMax(c, idx_max)
            c_max.append(cmin)
            print('max')

    return c_max
