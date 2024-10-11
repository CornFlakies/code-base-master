# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:48:22 2024

@author: coena
"""

# import cv2 as cv
import numpy as np
import skimage as sk
from skimage import measure
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy.optimize import curve_fit


@staticmethod
def Gx():
    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])
@staticmethod
def Gy():
    return np.array([[1, 2, 1],
                     [0, 0, 0],
                     [-1, -2, 1]])

@staticmethod
def matrix_convolve(matrix, kernel):
    '''
    Routine used to take same-size matrix and kernel, and computes the convolution
    of the center pixel of the provided matrix
    '''
    


@staticmethod
def gaussian(x, a, b, c):
    return a*np.exp(-(x - b)**2/(2*c**2)) 

def canny_edges(image):
    '''
    Routine to compute the Canny edges of the image.
    '''
    # Hard coding canny params. Provided a normalized image is used, defaults are:
    # t_min = 0.1 ; t_max = 0.2
    t_min = 0.1
    t_max = 0.2
    kernel = 3
    
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
    # analytical x-axis
    xana = np.linspace(0, 1, 50)
    
    # padding is taken from the left and right of the images
    padding = image.shape[0] // 10 #px

    for coord in edge_coords:
        x = coord[1]
        y = coord[0]
        image_row = image[y-padding:y+padding, x]
        edge_kernel = image[y-1:y+2, x-1:x+2]
        plt.figure()
        plt.imshow(edge_kernel)
        
        dir_x = matrix_convolve(edge_kernel, Gx())
        print(dir_x)
        
        grad = np.gradient(np.abs(image_row - 1))
        params, _ = curve_fit(gaussian, np.arange(0, len(grad)), grad)
        
        xana = np.linspace(0, len(grad), 50)
        yana = gaussian(xana, params[0], params[1], params[2])
        
        plt.figure()
        plt.plot(grad)
        plt.plot(xana, yana)
        plt.plot(params[1], params[0])
        print('peak loc: ' + str(params[1]))
        print('peak max: ' + str(params[0]))
        break
    return None

def detect_edges(image):
    '''
    Takes a single (8-bit, grayscale) image and detects the edge(s) of the droplets.
    This is done using subpix accuracy, consequently the algorithm spits out a vector
    of (x,y) coordinates of the edges.
    '''
    
    # Gaussian blurring
    image = sk.filters.gaussian(image, sigma=3)

    # Normalize to 1
    image = ((image - image.min()) / image.max())

    # Canny edges
    edges = canny_edges(image)
    
    # Get edge coordinates
    coords = np.column_stack(np.where(edges > 0))

    # Get subpixel accuracy on the edges
    coords_subpix = canny_subpix(image, coords)    

    return coords_subpix

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
    
    x = coords_edges[:, 0]
    y = coords_edges[:, 1]    
    
    # Second Splining method
    idxsort = np.argsort(x)
    # x = x[idxsort]
    # x = x[:-1]
    # y = y[idxsort]
    # y = y[:-1]
    
    # Create Spline and its derivative
    spline = scipy.interpolate.Rbf(x, y)
    print('jo')
    spline_deriv = spline.derivative()
    
    # Get maximum
    x_max = spline_deriv.roots()
    y_max = spline(x_max)
    xsp = np.linspace(x[0], x[-1], int(1E6))
    
    # Plot spline
    plt.figure()
    plt.scatter(x, y)
    plt.plot(xsp, spline(xsp), '-', color='red')
    
    # Plot the derivative
    plt.figure()
    plt.plot(xsp, spline_deriv(xsp))
    
    return x_max, y_max
    
    
        