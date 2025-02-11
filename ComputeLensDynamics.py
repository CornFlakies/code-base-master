# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 09:40:15 2024

@author: coena
"""
import os
import numpy as np
from tqdm import tqdm
import DetectSideView as dsv
import DetectTopView as dtv
import HelperFunctions as hp
import matplotlib.pyplot as plt


class ComputeLensDynamics:
    def __init__(self, input_dir, XMIN=0, XMAX=None, YMIN=0, YMAX=None, framestart=('stack 1', 0), view='side'):        
        print('\nInitializing...')
        if (view == 'side'):
            self.dv = dsv
        elif (view == 'top'):
            self.dv = dtv
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
    
    def get_R(self):
        '''
        Routine which is ran to get the side or top view height over time
        '''
        
        print('\nComputing R ...')
        # ------------------------ Extract the edges --------------------------
        # Run through all the tiff stacks, works for just one stack too
                    
        # Define list to store maxima
        r_max = []
        
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
            
            # Splines list
            self.splines = []
            
            # Run through the tiff stack
            for ii in tqdm(iterator):
                # Get image from stack
                image = hp.load_from_stack(stack_path, ii)[self.YMIN:self.YMAX+1, self.XMIN:self.XMAX+1]
                
                # Detects edges with subpixel accuracy
                coords_subpix = self.dv.detect_edges(image)
                
                try:
                    # Get the maximum from the coordinates (by fitting a spline)
                    c_max = self.dv.find_edge_extrema(image, coords_subpix)
                    
                    # Log r_max
                    r_max.append(c_max)
                    
                except Exception as exc: 
                    # Dump error message to standard output
                    print('\nException: ' + str(exc))
                    break
                
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
        
    