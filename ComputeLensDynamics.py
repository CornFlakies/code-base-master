# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 09:40:15 2024

@author: coena
"""
import os
import numpy as np
from tqdm import tqdm
import DetectDroplets as dp
import HelperFunctions as hp
import matplotlib.pyplot as plt

class ComputeLensDynamics:
    def __init__(self, input_dir, XMIN=None, XMAX=None, YMIN=None, YMAX=None, edge_detection='canny'):
        print('\nInitializing...')
        
        print('\nGetting files ...')
        self.image_paths, _ = hp.load_files(input_dir)
        print(f'     found {len(self.image_paths)} stack(s)')
        
        print('\nFinding frames of each tiff stack ...')
        self.stack_list = {}
        name = "Stack"
        for ii, path in enumerate(self.image_paths):
            self.stack_list['stack ' + str(ii + 1)] = ([0, hp.get_stack_size(path)], path)
            print(f'     stack {ii + 1}/{len(self.image_paths)} contains {hp.get_stack_size(path)} frames')
        
        print('\nFinding image dimensions ...')
        if self.image_paths is not None:
            self.image_shape = hp.read_from_stack(self.image_paths[0], 0).shape
        print(f'     image dimensions found: {self.image_shape[0]} x {self.image_shape[1]}')
        
        print('\nChecking window ...')
        self.check_window(XMIN, XMAX, YMIN, YMAX)
        
        print('\nFinding connecting frame ...')
        self.find_connecting_frame()
        
        print('all done!')
    
    def find_connecting_frame(self):
        '''
        function runs through all images in all stacks and finds where the 
        droplets first connect, and adjusts the stack_list accordingly
        '''
        
        # Iterate through stack
        for stack in self.stack_list:
            stack_data = self.stack_list[stack]
            print(stack_data)
            # Iterate through images
            
                # Find connecting frame
                    # Record frame, save to stack
                
                # If completely iterated through stack, delete stack from list
        
                
    
    def get_R(self):
        '''
        Big routine which is ran to get the side view height over time
        '''
        
        
        print('\nComputing R ...')
        
        # ----------------- Find Connecting Frame -----------------------------
        # Get the initial frame where the droplets connect
        stack_cnt = 0
        frame_cnt = 0
        for image_path, entries in zip(self.image_paths, self.frames):
            for entry in range(entries[1]):
                # Load image from tif stack
                image = hp.read_from_stack(image_path, entry)[self.YMIN:self.YMAX+1, self.XMIN:self.XMAX+1]
                
                # Break out of the loop if connection is found
                connected = dp.is_connected(image)
                if connected:
                    frame_start = frame_cnt
                    break
                else:
                    frame_cnt += 1
                    
            # Make sure to break out of the entire loop
            if connected:
                break
            else:
                stack_cnt += 1

        # Set the starting frame, at the appropriate stack
        frame_end = self.frames[stack_cnt][-1]
        self.frames[stack_cnt] = [frame_start, frame_end]
    
    
        # -------------------------- Extract the edges --------------------------------
        # Run through all the tiff stacks, works for just one stack too
        stack_tracker = 1
        diff = 10 #px
        r_max = []
        for image_path, entries in zip(self.image_paths[stack_cnt:], self.frames[stack_cnt:]):
            print(f"stack {stack_tracker}/{len(self.image_paths)}")    
            iterator = range(entries[0], entries[1])
            
            # Run through the tiff stack
            for ii, entry in tqdm(enumerate(iterator)):
                
                # Get image from stack
                image = hp.read_from_stack(image_path, entry)[self.YMIN:self.YMAX+1, self.XMIN:self.XMAX+1]
                image = (image / image.max() * 255).astype(np.uint8)
                
                # Detects edges with subpixel accurac
                coords_subpix = dp.detect_edges(image)
                
                # Get the maximum from the coordinates (by fitting a spline)
                x_max, y_max = dp.find_edge_extrema(coords_subpix)
                
                # Append the maxima
                r_max.append([x_max, y_max])
                
                break
                
                # Check if the maximum hasn't started moving too far.
                if ((np.abs(x_max - r_max[ii - 1][0]) > diff) & (ii != 0)):
                    r_max.pop()
                    print('\nIteration halted at frame ' + str(entry))
                    break
            # Iterate the stack tracker
            stack_tracker += 1
            break
        return r_max
    
    def check_window(self, XMIN, XMAX, YMIN, YMAX):
        '''
        Ugly ass if-else statements to determine if the input is defined or not.
        And if not, default to min or max image size. If defined, but incorrectly, it throws
        an error
        '''
        if (XMIN is None):
            print('XMIN not set, default to 0 ...')
            self.XMIN = 0
        elif (XMIN >= 0) & (XMIN < self.image_shape[1]):
            self.XMIN = XMIN
        else:
            raise Exception(f'ERROR: XMIN value of {XMIN} is invalid')
        if (XMAX is None):
            print('XMAX not set, default to max image size ...')
            self.XMAX = self.image_shape[1]
        elif (XMAX >= 0) & (XMAX < self.image_shape[1]):
            self.XMAX = XMAX
        else:
            raise Exception(f'ERROR: XMAX value of {XMAX} is invalid')
        if (YMIN is None):
            print('YMIN not set, default to 0 ...')
            self.YMIN = 0
        elif (YMIN >= 0) & (YMIN < self.image_shape[0]):
            self.YMIN = YMIN
        else:
            raise Exception(f'ERROR: YMIN value of {YMIN} is invalid')
        if (YMAX is None):
            print('YMAX not set, default to max image size ...')
            self.YMAX = self.image_shape[0]
        elif (YMIN >= 0) & (YMIN < self.image_shape[0]):
            self.YMAX = YMAX
        else:
            raise Exception(f'ERROR: YMAX value of {YMAX} is invalid')
        if (XMAX < XMIN) | (YMAX < YMIN):
            raise Exception('ERROR: Invalid values provided, either XMIN or YMIN is bigger than their respective MAX values')
    