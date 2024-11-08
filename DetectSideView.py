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
    padding = image.shape[1] // 6

    for ii, coord in enumerate(edge_coords):
        # Separate x- and y-coordinates for readability
        x = coord[0]
        y = coord[1]
        
        # Get region of interest from the image row
        image_row = image[y-padding:y+padding, x]
        
        # Compute gradient
        grad = np.abs(np.gradient(image_row))
        params, _ = curve_fit(gaussian, np.arange(0, len(grad)), grad)

        # Store subpixel edge to array
        coords_subpix[ii] = [coord[0], coord[1]  - padding + params[1]]

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

    return coords_subpix

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
    

def find_edge_extrema(image, coords_edges):
    '''
    Finds the maximum location based on the subpixel detected edge, a polynomial
    containing a 2nd and 4th order are fit to the data, and the resultant maximum
    from that polynomial is used to find the maxima in x and y
    '''
        
    def poly(x, x0, a, b, c):
        return a + b*(x - x0)**2 + c*(x - x0)**4
    
    # Separate x- and y-coordinates for readability
    x = coords_edges[:, 0]
    y = coords_edges[:, 1]    
    
    # Fit 2nd and 4th order polynomial
    p0 = [x[len(x)//2], 1, 1, 1]
    params, pcov = curve_fit(poly, x, y, p0)
    x_ana = np.linspace(x[0], x[-1], 50)

    x_max = params[0]
    y_max = poly(params[0], *params)
    
    return x_max, y_max


    # Sobel gradient stuff that I will probably not use
    def Gx():
        return np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    def Gy():
        return np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, 1]])
    
    def matrix_convolve(matrix, kernel):
        '''
        Routine takes same-size matrix and kernel, both the matrix and kernel
        are symmetric, and contain odd-numbered axes.
        Then, it computes the convolution of the center pixel of the provided matrix
        '''
        acc = 0
        for mat_val, ker_val in zip(matrix.flatten(), kernel.flatten()):
            acc += mat_val * ker_val
        return acc
    
    
        