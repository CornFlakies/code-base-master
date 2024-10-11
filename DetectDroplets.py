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
    # Hard coding canny params.
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
    
    # Remove duplicates on the x-axis, as they sample the same edge
    _, unique_indices = np.unique(edge_coords[:, 0], return_index=True)
    edge_coords = edge_coords[unique_indices]
    
    # Pre-allocate new array
    coords_subpix = np.zeros(edge_coords.shape, dtype=float)
    
    # padding is taken from the left and right of the images
    padding = image.shape[1] // 8 

    for ii, coord in enumerate(edge_coords):
        # Separate x- and y-coordinates for readability
        x = coord[0]
        y = coord[1]
        
        # Get region of interest from the image row
        image_row = image[y-padding:y+padding, x]

        # Compute gradient
        grad = np.gradient(np.abs(image_row - 1))
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
    # Gaussian blurring
    image = sk.filters.gaussian(image, sigma=3)

    # Normalize to 1
    image = ((image - image.min()) / image.max())

    # Get Canny edges
    edges = canny_edges(image)
    
    # Convert edges to coordinates
    coords = np.column_stack(np.where(edges.T > 0))

    # Get subpixel accuracy on the edges along the y-axis
    coords_subpix = canny_subpix(image, coords)    

    return coords_subpix, coords

def is_connected(image):
    '''
    Find the first frame where the droplets connect, such that a starting frame
    can be defined, from which to base the initial bridge height off.
    '''
    # Get canny edges
    edges = canny_edges(image)
    
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

def find_edge_extrema(coords_edges):
    '''
    Finds the maximum location based on the subpixel detected edge, a parameterization
    scheme is used to prepare the x- and y-data for cubic spline 
    interpolation (such that the data scales monotonically). The resultant spline \
    is then used to get the maximum location of the droplet bridge
    '''
        
    def poly(x, x0, a, b, c):
        return a + b*(x - x0)**2 + c*(x - x0)**4
    
    def poly_deriv(x, x0, b, c):
        return 2 * b * (x - x0) + 4*c*(x - x0)**3
    
    # Separate x- and y-coordinates for readability
    x = coords_edges[:, 0]
    y = coords_edges[:, 1]    
    
    # Fit 2nd and 4th order polynomial
    p0 = [x[len(x)//2], 1, 1, 1]
    print(p0)
    params, pcov = curve_fit(poly, x, y, p0)
    x_ana = np.linspace(x[0], x[-1], 50)
    
    plt.figure()
    # plt.plot(x, y, 'o', color='red')
    plt.plot(x_ana, poly(x_ana, *params), '-', color='black')
    
    # Create Spline and its derivative
    spline = CubicSpline(x, y)
    spline_deriv = spline.derivative()
    
    # plt.figure()
    # plt.plot(xsp, spline.derivative()(xsp), '-', color='blue')
    
    # Get maximum
    x_extrema = spline_deriv.roots()
    y_extrema = spline(x_extrema)
    x_max = x_extrema[(x_extrema > 0) & (x_extrema < x_extrema[-1])][0]
    y_max = y_extrema[(x_extrema > 0) & (x_extrema < x_extrema[-1])][0]
    
    plt.plot(x_ana, spline(x_ana), '-', color='blue')
    
    return x_max, y_max, spline

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
        Routine used to take same-size matrix and kernel, both the matrix and kernel
        are symmetric, and contain odd-numbered axes.
        Then, it computes the convolution of the center pixel of the provided matrix
        '''
        acc = 0
        for mat_val, ker_val in zip(matrix.flatten(), kernel.flatten()):
            acc += mat_val * ker_val
        return acc
    
    
        