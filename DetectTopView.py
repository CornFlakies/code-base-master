# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:05:45 2024

@author: coena
"""

import numpy as np
import skimage as sk
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from skimage.measure import find_contours


def connectedEdges(c):
    '''
    Check if the edges of the contour are connected
    '''
    if (np.linalg.norm(c[0] - c[-1]) < 10):
        return True

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
    Contour finding already ensures subpixel accuracy, do be careful though, 
    inhomogeneous background illumination messes with the contour finding algorithm
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
        return True
    else:
        return False

def find_edge_extrema(image, contours):
    '''
    Finds the maximum location based on the subpixel detected edge. The resultant spline \
    is then used to get the maximum location of the droplet bridge
    '''
    
    # Get the padding of the positively curved line around the maximum
    def getPaddingMax(c, idx_max):
        pad = 0
        ii = idx_max
        # print(c[:, 1])
        # Find the strictly decreasing points
        while ii < (len(c[:, 1]) - 1):
            if (c[ii, 1]) > (c[ii + 1, 1]):
                pad += 1
                ii += 1
            else:
                break
        pad_f = pad
        pad = 0
        ii = idx_max
        # Find the strictly decreasing points
        while ii > 0:
            if (c[ii, 1] < c[ii - 1, 1]):
                pad += 1
                ii -= 1
            else:
                break
        pad_b = pad
        return pad_f, pad_b
    
    # Get the padding of the positively curved line around the maximum
    def getPaddingMin(c, idx_max):
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
        # Find the strictly increasing points
        while ii > 0:
            if (c[ii, 1] > c[ii - 1, 1]):
                pad += 1
                ii -= 1
            else:
                break
        pad_b = pad
        return pad_f, pad_b
        
    
    # Get extremum of curved line
    def getMax(c, idx_max, ext):
        def poly(x, x0, A, B, C):
            return A + B * (x - x0)**2
        
        def semicircle(x, x0, y0, r):
            return y0 + np.sqrt(r - (x - x0)**2)
        
        if (ext == 'max'):
            pad_f, pad_b = getPaddingMax(c, idx_max)
        elif (ext == 'min'):
            pad_f, pad_b = getPaddingMin(c, idx_max)
        
        x = c[(idx_max - pad_b):(idx_max + pad_f), 1]
        y = c[(idx_max - pad_b):(idx_max + pad_f), 0]
        x_ana = np.linspace(x[0], x[-1], 100)
        
        spline_xmax = None
        spline_ymax = None
        
        y = y[np.argsort(x)]
        x = x[np.argsort(x)]
        spline = CubicSpline(x, y)
        spline_xmax = spline.derivative().roots()
        spline_ymax = spline(spline_xmax)
        
        if len(spline_xmax) > 1:
            closest_max = np.abs(spline_ymax - c[idx_max, 0])
            idx = np.argmin(closest_max)
        else:
            idx = 0
        
        spline_xmax = spline_xmax[idx]
        spline_ymax = spline_ymax[idx]
        return spline_xmax, spline_ymax
    
    # Log the maxima of the two contours
    c_max = []
    
    plt.imshow(image,cmap='gray')
    
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
            idx_ext = np.argmax(c[:, 0])
            cmax = getMax(c, idx_ext, ext='max')
            c_max.append(cmax)
            plt.plot(c[:, 1], c[:, 0], color='red')
            plt.plot(cmax[0], cmax[1], 'o', color='blue')
        # If contour below midline
        elif all(cb) and not all(ct):
            idx_ext = np.argmin(c[:, 0])
            cmin = getMax(c, idx_ext, ext='min')
            c_max.append(cmin)
            plt.plot(c[:, 1], c[:, 0], color='red')
            plt.plot(cmin[0], cmin[1], 'o', color='blue')
    return c_max
