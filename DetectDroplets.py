# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:48:22 2024

@author: coena
"""

import cv2 as cv
import matplotlib.pyplot as plt

def detect_edges(image):
    '''
    Takes a single (8-bit, grayscale) image and detects the edge(s) of the droplets 
    '''
    
    edges = cv.Canny(image, 50, 150)
    # Display the original image and edges
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Grayscale Image')

    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap='gray')
    plt.title('Edge Detection')