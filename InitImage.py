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
from ImageVisualizer import ImageVisualizer
from CalibrateImages import CalibrateImages


def create_config_from_suite(abs_path, fps):
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
            init_view_img = sk.io.imread(path_init_view_img[0])
            radii, centers = get_drop_radii(init_view_img)
            
            # Plot droplets and save image
            plot_drops(radii, 
                       centers, 
                       init_view_img, 
                       alpha_top, 
                       path=os.path.join(abs_path, directory),
                       show=False)
            
            # Convert Radius
            R = np.mean(radii)
            R *=  alpha_top
            
            # User select starting frame from the batch of measurements
            # Do a check if config file alread exists
            path_dir = os.path.join(abs_path, directory)
            path_config_file, config_file = hp.load_files(path_dir, header="yaml")
            
            if (config_file != []):
                with open(path_config_file[0], 'r', encoding='utf-8') as file:
                    data = yaml.safe_load(file)
                    print(yaml.dump(data, default_flow_style=False))

                print(f'config file already exists in dir {directory}, make a new one?\n')
                Flag = False
                while not Flag:
                    boolean = doConfigCreation()
                    if boolean is None:
                        continue
                    else:
                        Flag = True

            if boolean:
                # Top view
                path_top_view  = os.path.join(abs_path, directory, 'top_view')
                top_image_paths, _ = hp.load_files(path_top_view, 'tif')
                iv = ImageVisualizer(top_image_paths[0], view='top')
                start_frame_top, manual_points_top = iv.get_data()
                # Dump manual points into .npy file
                if (np.sum(~np.isnan(manual_points_top[:, 0])) != 0):
                    np.save(os.path.join(abs_path, directory, 'manual_points_top.npy'), manual_points_top)
                
                # Side view
                path_side_view = os.path.join(abs_path, directory, 'side_view') 
                side_image_paths, _ = hp.load_files(path_side_view, 'tif')
                iv = ImageVisualizer(side_image_paths[0], view='side')
                start_frame_side, manual_points_side = iv.get_data()
                # Dump manual points into .npy file
                if (np.sum(~np.isnan(manual_points_side[:, 0])) != 0):
                    np.save(os.path.join(abs_path, directory, 'manual_points_side.npy'), manual_points_side)
                
                # # Dump everything into a yaml file for future data analysis
                # config_data = {
                #     "MEASUREMENTS_PARAMETERS": {
                #         "DROP_RADIUS": float(R),
                #         "FPS": float(fps),
                #         "CONV_TOP_VIEW": float(alpha_top),
                #         "CONV_SIDE_VIEW": float(alpha_side),
                #         },
                #     "INITIAL_PARAMETERS": {
                #         "INITIAL_FRAME_TOP_VIEW": int(start_frame_top),
                #         "INITIAL_FRAME_SIDE_VIEW": int(start_frame_side)
                #         }
                #     }
                
                
                # with open(os.path.join(abs_path, directory, 'config.yaml'), 'w', encoding="utf-8") as file:
                #     file.truncate()
                #     yaml.dump(config_data, file, default_flow_style=False)

def get_drop_radii(image):
    '''
    Gets radii from the droplets in view, spits out the mean of the radii
    '''
    # Load image from path, and normalize
    img = hp.normalize_img(image)
    
    # Get contours from normalized image
    contours = find_contours(img)
    radii = []
    centers = []
    for c in contours:
        a, b, R = circular_regression_bullock(c)
        a = a[0]
        b = b[0]
        R = R[0]
        radii.append(R)
        centers.append([a, b])
    
    return radii, centers

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
                   
def plot_drops(radii, centers, img, alpha_top, path='', show=True):
    
    radii = np.asarray(radii, dtype=float)
    centers = np.asarray(centers, dtype=float)
    
    # Define circles
    circle1 = {"center": (centers[0][0], centers[0][1]), "radius": radii[0]}
    circle2 = {"center": (centers[1][0], centers[1][1]), "radius": radii[1]}
    
    circle1_patch = plt.Circle(circle1["center"], circle1["radius"], color='blue', lw=2, fill=False, label='Circle 1')
    circle2_patch = plt.Circle(circle2["center"], circle2["radius"], color='red', lw=2, fill=False, label='Circle 2')
    
    mean_center_x = (centers[0][0] + centers[1][0]) / 2
    mean_center_y = (centers[0][1] + centers[1][1]) / 2
    
    # Add circles to the plot
    fig, ax = plt.subplots()
    # Set title
    plt.title('Two (least squares) Circles Plot')
    # Set x- and y-labels
    ax.set_xlabel('X [px]')
    ax.set_ylabel('Y [px]')
    
    # Get maximum radius
    radius = max(radii[0], radii[1])
    
    # Plot image
    ax.imshow(img, cmap='gray')
    
    # Plot circles
    ax.add_patch(circle1_patch)
    ax.add_patch(circle2_patch)
    
    # Save figure
    path1 = os.path.join(path, 'init_circles_zoom.png')
    plt.savefig(path1)
    
    # Create new limits for second figure
    ax.set_xlim([-radius*2.5 + mean_center_x, radius*2.5] + mean_center_x)
    ax.set_ylim([-radius*2.5 + mean_center_y, radius*2.5] + mean_center_y)
    
    # Draw an arrow for the radius from the center1 to the edge
    center = centers[0]
    radius = radii[0]
    color = 'blue'
    
    theta = np.pi / 2  # Angle for the radius line (can be any value)
    arrow_x = center[0] + radius * np.cos(theta)
    arrow_y = center[1] + radius * np.sin(theta)
    
    # Plot the arrow
    ax.annotate(
        "", xy=(arrow_x, arrow_y), xytext=center,
        arrowprops=dict(arrowstyle="->", color=color)
    )
    
    # Label the radius
    midpoint = (1.5 * (center[0] + arrow_x) / 2, 2 * (center[1] + arrow_y) / 2)
    radius_text = radius * alpha_top * 1e3
    ax.text(midpoint[0], midpoint[1], rf"r $\approx$ {radius_text:0.2f} $mm$", color=color, fontsize=10)
    
    # Draw an arrow for the radius from the center2 to the edge
    center = centers[1]
    radius = radii[1]
    color = 'red'
    
    theta = np.pi / 2  # Angle for the radius line (can be any value)
    arrow_x = center[0] + radius * np.cos(theta)
    arrow_y = center[1] + radius * np.sin(theta)
    
    # Plot the arrow
    ax.annotate(
        "", xy=(arrow_x, arrow_y), xytext=center,
        arrowprops=dict(arrowstyle="->", color=color)
    )
    
    # Label the radius
    midpoint = (1.0 * (center[0] + arrow_x) / 2, 2 * (center[1] + arrow_y) / 2)
    radius_text = radius * alpha_top * 1e3
    ax.text(midpoint[0], midpoint[1], rf"r $\approx$ {radius_text:0.2f} $mm$", color=color, fontsize=10)
    
    # Save figure
    path2 = os.path.join(path, 'init_circles.png')
    plt.savefig(path2)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
def doConfigCreation():
    patterny = re.compile(r'\b(?:yes|y)\b', re.IGNORECASE)
    patternn = re.compile(r'\b(?:no|n)\b', re.IGNORECASE)
    command = input('[y/n]\n')
    if (patterny.search(command)):
        return True
    elif (patternn.search(command)):
        return False
    else:
        return None
    
    
    
