# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:23:00 2025

@author: coena
"""

import numpy as np
import HelperFunctions as hp

import matplotlib
matplotlib.use('Qt5Agg')  # Ensure interactive backend is selected
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Define location measurement suite
file_path = "D:\\masterproject\\images\\dodecane_17012025\\set2\\meas2\\dodecane_side_view.tif"

# Size of the 1st stack
stack_length = hp.get_stack_size(file_path)

class ImageVisualizer:
    def __init__(self, file_path):
        # Define general variables, buttons, plot and create image
        self.start_frame = 0
        self.file_path = file_path
        self.length = hp.get_stack_size(file_path)
        self.frame = 0
        self.exit = False
        
        # Create plot
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.3) 
        
        # Load initial image and show it, define image handle
        data = hp.load_from_stack(file_path, 0)
        self.im_handle = self.ax.imshow(data, cmap='gray')
        self.update_image()
    
        # Define button axes
        axprev10 = self.fig.add_axes([0.3, 0.1, 0.1, 0.075])
        axprev = self.fig.add_axes([0.4, 0.1, 0.1, 0.075])
        axnext = self.fig.add_axes([0.5, 0.1, 0.1, 0.075])
        axnext10 = self.fig.add_axes([0.6, 0.1, 0.1, 0.075])
        axset = self.fig.add_axes([0.8, 0.1, 0.1, 0.075])
        axclose = self.fig.add_axes([0.1, 0.1, 0.1, 0.075])
    
        # Create buttons and bind callbacks
        bnext = Button(axnext, '+1')
        bnext.on_clicked(self.next)
        
        bnext10 = Button(axnext10, '+10')
        bnext10.on_clicked(self.next10)
    
        bprev = Button(axprev, '-1')
        bprev.on_clicked(self.prev)
        
        bprev10 = Button(axprev10, '-10')
        bprev10.on_clicked(self.prev10)
        
        bclose = Button(axclose, 'Close')
        bclose.on_clicked(self.close)
        
        bset = Button(axset, 'Set')
        bset.on_clicked(self.set_start_frame)
        
        # Set up the close event to break the loop gracefully
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        # Explicitly show plot to keep figure active
        plt.show()
        
        # Keep the program running until the figure is closed
        while not self.exit:
            plt.pause(0.1)
    
    def next(self, event):
        self.frame = (self.frame + 1) % self.length
        self.update_image()
        
    def next10(self, event):
        self.frame = (self.frame + 10) % self.length
        self.update_image()

    def prev(self, event):
        self.frame = (self.frame - 1) % self.length
        self.update_image()
        
    def prev10(self, event):
        self.frame = (self.frame - 10) % self.length
        self.update_image()
        
    def set_start_frame(self, event):
        self.start_frame = self.frame
        self.update_image()
    
    def update_image(self):
        self.im_handle.set_data(hp.load_from_stack(self.file_path, self.frame))
        self.ax.set_title(f'frame {self.frame}, start frame {self.start_frame}')
        self.fig.canvas.draw()
    
    def close(self, event):
        plt.close(self.fig)
        
    def on_close(self, event):
        print('Closing ...')
        print(f'Start frame set at: {self.start_frame}')
        self.exit = True
    
plt.close('all')
iv = ImageVisualizer(file_path)
