# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:55:13 2024

@author: coena
@description: 
    Helperfunctions should be stored in this script so they can be 
    readily imported for other analysis scripts. Example helperfunctions
    are image loading, directory creating, directory reading etc.
"""

import os
import numpy as np
import skimage as sk

def get_stack_size(input_file):
    '''
    Gets the size of a tiff stack
    '''
    tiff_collection = sk.io.ImageCollection(input_file)
    return len(tiff_collection)

def load_from_stack(input_file, frame):
    '''
    Helper function loads an image from a tiff stack
    '''
    return sk.io.imread(input_file, img_num=frame)

def load_files(input_dir, header='tif'):
    '''
    Helper function used to get string array of all files with header 
    in supplied directory. Supply the header without the preceding dot
    '''
    
    # Load in image paths
    images = np.sort(os.listdir(input_dir))
    
    # find npy files
    image_paths = []
    image_names = []
    for entry in images:
        split_filename = entry.split('.')
        if (split_filename[-1] == header):
            image_names.append(entry)
            image_paths.append(os.path.join(input_dir, entry))
            
    return np.sort(image_paths), np.sort(image_names)

def create_output_dir(output_folder):
    '''
    Helper function used to generate an output folder for the processed data
    '''
    if not os.path.isdir(output_folder):
        print(f'Directory {output_folder} does not exist, making directory ...')
        os.makedirs(output_folder)
    elif any([('.' in item) for item in os.listdir(output_folder)]):
        raise Exception(f'Directory {output_folder} already exists, and contains files, check that you do not overwrite anything!')
    else:
        print(Warning(f'Directory {output_folder} already exists, but does not contain files, so nothing will get overwritten, continuing ..'))

def print_directory_tree(input_dir, indent_level=0):
    for root, dirs, files in os.walk(input_dir):
        # Determine the current level by counting the number of separators in the root
        level = root.replace(input_dir, '').count(os.sep)
        
        # If the current level is greater than the desired indent_level, skip to the next iteration
        if level > indent_level:
            continue
        
        # Indent the folder by the level number, add a branch symbol
        indent = ' ' * 4 * (level)
        print(f"{indent}|-- {os.path.basename(root)}/")
        
        # Indent the files by one more level than the folder
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}|-- {f}")

def get_dir_tree(input_dir):
    return None
    