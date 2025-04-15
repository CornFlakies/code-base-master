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

def detect_edges(image):
    '''
    Takes a single (8-bit, grayscale) image and detects the edge(s) of the droplets.
    Contour finding already ensures subpixel accuracy, do be careful though, 
    inhomogeneous background illumination messes with the contour finding algorithm
    '''
    # Get contours
    contours = contour_edges(image)
    
    return contours

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
    
    # Find midline, such that contours above and below can be identified
    com = []
    if len(contours) > 1:
        for c in contours:
            com.append(np.mean(c[:, 0]))
    midline = np.sum(com) / len(com)
    
    def is_contour_closed(contour, tol=1e-6):
        """Check if the contour is closed by comparing first and last point."""
        return np.linalg.norm(contour[0] - contour[-1]) < tol
    
    relevant_contours = []
    for ii, c in enumerate(contours):
        if is_contour_closed(c, tol=1e-3):
            continue
        else:
            relevant_contours.append(c)
            
    print(len(contours))
    print(len(relevant_contours))
    for ii, c in enumerate(relevant_contours):
        ct = c[:, 0] < (midline + 20)
        cb = c[:, 0] > (midline - 20)
        # Remove the circular contour crossing the midline
        if (any(cb) and any(ct)):
            continue
        # If contour above midline
        if all(ct) and not all(cb):
            idx_ext = np.argmax(c[:, 0])
            cmax = getMax(c, idx_ext, ext='max')
            # c_max.append(cmax)
        # If contour below midline
        elif all(cb) and not all(ct):
            idx_ext = np.argmin(c[:, 0])
            cmin = getMax(c, idx_ext, ext='min')
            c_max.append(cmin)
    return c_max
