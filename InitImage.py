# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:46:05 2025

@author: coena
"""

import os
import re
import yaml
import numpy as np
import skimage as sk
import HelperFunctions as hp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage.measure import find_contours
from CalibrateImages import CalibrateImages

def create_config_from_suite(abs_path, fps):
    # Define FPS of the measurements
    fps = 1e5
    
    # Load the files and directories from path
    contents = os.listdir(abs_path)
    
    # Get images for calibration, define alpha of both top and side view
    calib_paths, calib_files = hp.load_files(abs_path, header="tif")
    ci = CalibrateImages()
    
    # Multiple calibration images can be provided
    alpha_top = []
    alpha_side = []
    for path, file in zip(calib_paths, calib_files):
        view = re.search("top|side", file).group()
        dist = ci.run(path, file)
        if (view == "top"):
            alpha_top.append(dist)
        elif (view == "side"):
            alpha_side.append(dist)
        
    # Define conversion factors for the top and side view
    alpha_top = np.mean(alpha_top)
    alpha_side = np.mean(alpha_side)
    
    # For each measurement, compute the initial radius of the droplets
    for directory in contents:
        if bool(re.search("meas", directory)):
            
            # Load init view images and get drop radii
            path_init_view = os.path.join(abs_path, directory, 'init_top_view')
            path_init_view_img, init_view_img = hp.load_files(path_init_view, header="tif")
            R = get_drop_radii(path_init_view_img[0]) * alpha_top
            
            # User select starting frame from the batch of measurements
            
            
            # Dump everything into a yaml file for future data analysis
            config_data = {
                "MEASUREMENTS_PARAMETERS": {
                    "FPS": float(fps),
                    "CONV_TOP_VIEW": float(alpha_top),
                    "CONV_SIDE_VIEW": float(alpha_side),
                    },
                "INITIAL_PARAMETERS": {
                    "DROP_RADIUS": float(R),
                    "INITIAL_FRAME": -1
                    }
                }
            
            with open(os.path.join(abs_path, directory, 'config.yaml'), 'w') as file:
                yaml.dump(config_data, file, default_flow_style=False)

def define_starting_frame(path_to_images):
    image_paths, images = hp.load_files(path_to_images, header='tif')
    
    

def get_drop_radii(image):
    '''
    Gets radii from the droplets in view, spits out the mean of the radii
    '''
    # Load image from path, and normalize
    img = sk.io.imread(image)
    img = hp.normalize_img(img)
    
    # Get contours from normalized image
    contours = find_contours(img)
    radii = []
    for c in contours:
        a, b, R = circular_regression_bullock(c)
        radii.append(R)
    R = np.mean(radii)
    
    return R

def circular_regression_bullock(c):
    '''
    Circular regression method takes points on part of an arc of a circle and 
    finds the least squares fit with minimzation equation: 
        
            g(u, v) = (u - u_c)^2 + (v - v_c)^2 - alpha
            
    such that for the set of points (u_1, v_1), ... (u_i, v_i), ... (u_n, v_n):
        
            sum over i to n (g(u_i, v_i)) = 0
            
    with n the amount of data points and:
        
            u = x - xhat
            v = y - yhat
            alpha = R^2
         
    with xhat and yhat being the mean values of x and y

    The full derivation can be found on: 
    https://dtcenter.org/sites/default/files/community-code/met/docs/write-ups/circle_fit.pdf    
      
    '''
    
    def S(nu, nv):
        return np.sum(u**nu * v**nv)
    
    # Define x- and y-points spaning an arc of the circle-to-betermined
    n = float(len(c))
    x = c[:, 1]
    y = c[:, 0]
    
    # Create mean values
    x_hat = np.mean(x)
    y_hat = np.mean(y)
    
    # Define u and v as x and y with zero mean
    u = x - x_hat
    v = y - y_hat
    
    # Prepare arrays for the solving of least squares matrix
    m1 = np.array([[S(2, 0), S(1, 1)],
                   [S(1, 1), S(0, 2)]])
    
    m2 = np.array([[S(3, 0) + S(1, 2)],
                   [S(0, 3) + S(2, 1)]])
    
    # Do computation
    ucvc = 1/2 * np.linalg.inv(m1) @ m2
    
    # Get alpha
    alpha = ucvc[0]**2 + ucvc[1]**2 + (S(2, 0) + S(0, 2)) / n
    
    # Return values a, b and R, transforming them back to original coordinates
    return ucvc[0] + x_hat, ucvc[1] + y_hat, np.sqrt(alpha)
                   
    
    
    
    
    
