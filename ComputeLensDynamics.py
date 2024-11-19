# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 09:40:15 2024

@author: coena
"""
import os
import numpy as np
from tqdm import tqdm
import DetectSideView as dps
import DetectTopView as dpt
import HelperFunctions as hp
import matplotlib.pyplot as plt


class ComputeLensDynamics:
    def __init__(self, input_dir, XMIN=0, XMAX=None, YMIN=0, YMAX=None, framestart=('stack 1', 0), view='side', edge_detection='canny'):        
        print('\nInitializing...')
        if (view == 'side'):
            self.dp = dps
        elif (view == 'top'):
            self.dp = dpt
        else:
            raise Exception('Invalid view argument')
        
        print('\nGetting files ...')
        self.image_paths, _ = hp.load_files(input_dir)
        print(f'     found {len(self.image_paths)} stack(s)')
        
        print('\nFinding frames of each tiff stack ...')
        self.stack_list = {}
        for ii, path in enumerate(self.image_paths):
            self.stack_list['stack ' + str(ii + 1)] = ([0, hp.get_stack_size(path)], path)
            print(f'     stack {ii + 1}/{len(self.image_paths)} contains {hp.get_stack_size(path)} frames')
        
        print('\nFinding image dimensions ...')
        if self.image_paths is not None:
            self.image_shape = hp.load_from_stack(self.image_paths[0], 0).shape
        print(f'     image dimensions found: {self.image_shape[0]} x {self.image_shape[1]}')
        
        print('\nChecking window ...')
        self.check_window(XMIN, XMAX, YMIN, YMAX)
        
        if (framestart == ('stack 1', 0)):
            print('\nFinding connecting frame ...')
            self.find_connecting_frame()
        else:
            print(f'\nStarting stack and frame specified: {framestart[0]}, frame {framestart[1]}')
            self.define_starting_frame(framestart)
        
        print('all done!')
    
    def define_starting_frame(self, framestart):
        '''
        Routine used to set the starting frame and starting stack
        '''
        stack_to_pop = []
        for stack in self.stack_list:
            
            # Get stack info
            stack_start = self.stack_list[stack][0][0]
            stack_end = self.stack_list[stack][0][1]
            stack_path = self.stack_list[stack][1]
            
            if (stack == framestart[0]):
                stack_start = framestart[1]
            else:
                stack_to_pop.append(stack)
                
        # Update stack
        self.stack_list.update({stack:[[stack_start, stack_end], stack_path]})
        
        # Delete the appended stacks
        if stack_to_pop:
            for stack in stack_to_pop:
                self.stack_list.pop(stack)     
    
    def find_connecting_frame(self):
        '''
        function runs through all images in all stacks and finds where the 
        droplets first connect, and adjusts the stack_list accordingly
        '''
        # Define list of stacks to pop
        stack_to_pop = []
        
        # Iterate through stack
        for stack in self.stack_list:
            
            # Get stack info
            stack_start = self.stack_list[stack][0][0]
            stack_end = self.stack_list[stack][0][1]
            stack_path = self.stack_list[stack][1]
            print(f"stack {stack}/{len(self.stack_list)}")  
            
            # Iterator to go through stack
            iterator = range(stack_start, stack_end)
            
            # Iterate through images
            for ii in iterator:
                image = hp.load_from_stack(stack_path, ii)[self.YMIN:self.YMAX+1, self.XMIN:self.XMAX+1]
                # Find connecting frame
                connected = self.dp.is_connected(image)
                if connected:
                    # Record frame, save to stack
                    # plt.figure()
                    # plt.imshow(image)
                    print(f'    found connection in {stack} frame {ii}!')
                    stack_start = ii
                    break
            
            # Make sure to break out of the entire loop
            if connected:
                break
            else:
                # Delete stack from dictionary if frame is not found
                stack_to_pop.append(stack)
        
        # Update stack
        self.stack_list.update({stack:[[stack_start, stack_end], stack_path]})
        
        # Delete the appended stacks
        if stack_to_pop:
            for stack in stack_to_pop:
                self.stack_list.pop(stack)       
    
    def get_R(self):
        '''
        Big routine which is ran to get the side view height over time
        '''
        
        print('\nComputing R ...')
        # ------------------------ Extract the edges --------------------------
        # Run through all the tiff stacks, works for just one stack too
        
        # Iterate through stack
        for stack in self.stack_list:
            
            # Get stack info
            stack_start = self.stack_list[stack][0][0]
            stack_end = self.stack_list[stack][0][1]
            stack_path = self.stack_list[stack][1]
            print(f"{stack}/{len(self.stack_list)}")    
            
            # Iterator to go through stack
            iterator = range(stack_start, stack_end)
            print(iterator)
            
            # Define list to store maxima
            r_max = []
            
            # Run through the tiff stack
            for ii in iterator:
                # Get image from stack
                image = hp.load_from_stack(stack_path, ii)[self.YMIN:self.YMAX+1, self.XMIN:self.XMAX+1]
                
                # Detects edges with subpixel accuracy
                coords_subpix = self.dp.detect_edges(image)
                
                x_max, y_max = self.dp.find_edge_extrema(image, coords_subpix)
                    
                # try:
                #     # Get the maximum from the coordinates (by fitting a spline)
                #     x_max, y_max = self.dp.find_edge_extrema(image, coords_subpix)
                #     r_max.append((x_max, y_max))
                #     # plt.plot(x_max[0], x_max[1], '.', color='green')
                #     # plt.plot(y_max[0], y_max[1], '.', color='red')
                #     break
                # except:
                #     break
        
        return r_max
    
    def check_window(self, XMIN, XMAX, YMIN, YMAX):
        '''
        Ugly ass if-else statements to determine if the input window is defined or not.
        And if not, default to min or max image size. If defined, but incorrectly, it throws
        an error
        '''
        if (XMIN == 0):
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
        if (YMIN == 0):
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
    