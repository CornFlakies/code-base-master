# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:48:22 2024

@author: coena
"""

import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline


def canny_edges(image):
    '''
    Routine to compute the Canny edges of an image.
    '''
    # Gaussian blurring
    image = sk.filters.gaussian(image, sigma=3)

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

def canny_subpix(image, edge_coords):
    '''
    Routine used to get supixel location of the edges. Uses the canny edge coords
    as point from which to get the gradient. The gradient is taken to be in the 
    direction parallel to the ground, or upwards/downards
    '''
    
    # Gaussian fucntion used for the fitting routine
    def gaussian(x, a, b, c):
        return a*np.exp(-(x - b)**2/(2*c**2)) 
    
    # Gaussian blurring
    image = sk.filters.gaussian(image, sigma=3)
    
    # Remove duplicates on the x-axis, as they sample the same edge
    _, unique_indices = np.unique(edge_coords[:, 0], return_index=True)
    edge_coords = edge_coords[unique_indices]
    
    # Pre-allocate new array
    coords_subpix = np.zeros(edge_coords.shape, dtype=float)
    
    # padding is taken from the left and right of the images
    padding = 40

    for ii, coord in enumerate(edge_coords):
        # Separate x- and y-coordinates for readability
        x = coord[0]
        y = coord[1]
        
        lower_pad = max(y - padding, 0)
        upper_pad = min(y + padding, image.shape[0])
        
        # Get region of interest from the image row
        image_row = image[lower_pad:upper_pad, x]
        
        # Compute gradient
        grad = np.abs(np.gradient(image_row))
        p0 = [np.max(image_row), len(image_row) / 2, 1] # make some guess params
        params, _ = curve_fit(gaussian, np.arange(0, len(grad)), grad, p0)
        
        # Store subpixel edge to array
        coords_subpix[ii] = [lower_pad + params[1], coord[0]]

    return coords_subpix

def detect_edges(image):
    '''
    Takes a single (8-bit, grayscale) image and detects the edge(s) of the droplets.
    To get supbixel accuracy, a Gaussian interpolation is done along the edge pixels
    detected by a Canny edge detector.
    '''
    
   # Get Canny edges
    edges = canny_edges(image)
    
    # Convert edges to coordinates
    coords = np.column_stack(np.where(edges.T > 0))

    # Get subpixel accuracy on the edges along the y-axis
    coords_subpix = canny_subpix(image, coords)    

    return [coords_subpix]

def find_edge_extrema(image, coords_edges):
    '''
    Finds the maximum location based on the subpixel detected edge, a polynomial
    containing a 2nd and 4th order are fit to the data, and the resultant maximum
    from that polynomial is used to find the maxima in x and y
    '''
    
    c_max = []
    for c in coords_edges:
        
        def poly(x, x0, a, b, c):
            return a + b*(x - x0)**2 + c*(x - x0)**4
        
        # Separate x- and y-coordinates for readability
        x = c[:, 1]
        y = c[:, 0]    
        
        # # Fit 2nd and 4th order polynomial
        # p0 = [x[len(x)//2], 1, 1, 1]
        # params, pcov = curve_fit(poly, x, y, p0)
    
        # # Save maximum location
        # x_ana = np.linspace(x[0], x[-1], 100)
        # x_max = params[0]
        # y_max = poly(params[0], *params)
        
        # fit spline 
        spline = CubicSpline(x, y, axis=0)
        spline_minima = spline.derivative().roots()
        
        # Extract root closest to the minimum value of the computed contour
        x_loc = x[np.argmax(y)]
        x_loc = spline_minima[np.argmin(np.abs(spline_minima - x_loc))]
        y_loc = spline(x_loc).item()
        c_max.append((x_loc, y_loc))

    return c_max
    
    
        