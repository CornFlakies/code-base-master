# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:48:22 2024

@author: coena
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from devernay_edges import DevernayEdges

def detect_edges(image, T_min=50, T_max=150, kernel=(5,5)):
    '''
    Takes a single (8-bit, grayscale) image and detects the edge(s) of the droplets 
    '''
    
    # Blur image and get edges
    blur = cv.GaussianBlur(image, kernel, 0)
    edges = cv.Canny(blur, T_min, T_max)
    return edges

def refine_edges(image, edges, buffer_size=5):
    '''
    Takes the (8-bit) edges object and creates a mask for the original image so that
    a better threshholding algorithm can be applied to the image. Then the mask
    is applied to the provided image (which should be the same size as the 
    edges array) and interpolates the edge to get subpixel accuracy.
    '''
    
    # Pre-allocated the buffered mask array, and the buffer to be placed
    buffered_mask = np.zeros(edges.shape)
    buffer = np.ones(buffer_size * 2)
    plt.figure()
    plt.imshow(image)
    # Look over all columns
    for ii in range(np.size(edges, axis=1)):
        # Get column and find the edge location
        col = edges[:, ii]
        edge_loc = np.argwhere(col==255)
        
        # If there are multiple pixels in the column choose the "middle" one
        # The exact location is not that important
        if (edge_loc==[]):
            pass
        if (len(edge_loc) > 1):
            edge_loc = edge_loc[len(edge_loc) // 2][0]
        else:
            edge_loc = edge_loc[0][0]
        
        image_edge = image[edge_loc-buffer_size:edge_loc+buffer_size, ii].astype(np.float64)
        image_edge /= max(image_edge)
        plt.figure()
        plt.plot(image_edge, '.-')
        # plt.plot(buffer_size, image[edge_loc, ii], '.', color='red')
        
        break