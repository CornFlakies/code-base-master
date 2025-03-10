# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:23:00 2025

@author: coena
"""

import numpy as np
import HelperFunctions as hp
import DetectTopView as dtv
import DetectSideView as dsv

import matplotlib
matplotlib.use('Qt5Agg')  # Ensure interactive backend is selected
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# # Size of the 1st stack
# stack_length = hp.get_stack_size(file_path)

class ImageVisualizer:
    '''
    Class takes in a tiff stack and allows to scan through the data set, setting
    a start frame on which coalescence starts, and allows selecting points for where 
    the edge finding algorithms do not detect the relevant bridge.
    '''
    def __init__(self, file_path, view='side'):
        
        # Load side view or top view finder
        if (view == 'side'):
            self.dv = dsv
        elif (view == 'top'):
            self.dv = dtv
        else:
            raise Exception(f'Check class initialization, invalid view argument: {view}')
            
        # Define general variables, buttons, plot and create image
        self.start_frame = 0
        self.file_path = file_path
        self.length = hp.get_stack_size(file_path)
        self.frame = 0
        self.manual_points = np.empty((self.length, 2))
        self.manual_points[:] = np.nan
        self.cont_handles = []
        self.exit = False
        self.setContours = False
        
        # Create plot
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.3)
        plt.subplots_adjust(left=-0.01)
        
        # Load initial image and show it, define image handle
        data = hp.load_from_stack(file_path, 0)
        self.im_handle = self.ax.imshow(data, cmap='gray')
        self.update_image()
    
        # Define button size
        bottom_button_size = [0.1, 0.1, 0.075]
    
        # Define button axes
        axprev10 = self.fig.add_axes([0.25, *bottom_button_size])
        axprev = self.fig.add_axes([0.35, *bottom_button_size])
        axnext = self.fig.add_axes([0.45, *bottom_button_size])
        axnext10 = self.fig.add_axes([0.55, *bottom_button_size])
        axclose = self.fig.add_axes([0.05, *bottom_button_size])
        axdel = self.fig.add_axes([0.85, 0.6, 0.1, 0.075])
        axreset = self.fig.add_axes([0.85, 0.5, 0.1, 0.075])
        axcontour = self.fig.add_axes([0.85, 0.4, 0.1, 0.075])
        axset = self.fig.add_axes([0.85, 0.3, 0.1, 0.075])
    
        # Create buttons and bind callbacks
        bnext = Button(axnext, '+1')
        bnext.on_clicked(self.next)
        
        bnext10 = Button(axnext10, '+10')
        bnext10.on_clicked(self.next10)
    
        bprev = Button(axprev, '-1')
        bprev.on_clicked(self.prev)
        
        bprev10 = Button(axprev10, '-10')
        bprev10.on_clicked(self.prev10)
        
        bdel = Button(axdel, 'Delete')
        bdel.on_clicked(self.del_point)
        
        breset = Button(axreset, 'Reset')
        breset.on_clicked(self.reset_points)
        
        bclose = Button(axclose, 'Close')
        bclose.on_clicked(self.close)
        
        bset = Button(axset, 'SetStart')
        bset.on_clicked(self.set_start_frame)

        bcontour = Button(axcontour, 'Contour')
        bcontour.on_clicked(self.toggle_contour)
        
        # Set up click event to store manually selected points
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Set up the close event to break the loop gracefully
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        # Display some messages
        print('Image visualiser initialised ...')
        print('Make sure to check that all manually selected points are before the selected start frame')
        print('If this is not the case, the selected points will not be saved!!')
        
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
        
    def del_point(self, event):
        self.manual_points[self.frame] = [np.nan, np.nan]
        self.update_image()
        print('Deleted current point ...')
    
    def reset_points(self, event):
        self.manual_points = np.empty((self.length, 2))
        self.manual_points[:] = np.nan
        self.update_image()
        print('Reset selected points ...')
        
    def on_click(self, event):
        if event.button == 3: # 3 is the right-click button
            ix, iy = event.xdata, event.ydata
            print(f'Selected point: [{ix}, {iy}]')
            self.manual_points[self.frame] = [ix, iy]
            self.update_image()
        
    def toggle_contour(self, event):
        self.setContours ^= True
        self.update_image()
                
    def update_image(self):
        # Load image
        image = hp.load_from_stack(self.file_path, self.frame)
        
        # Remove old contours
        if (self.cont_handles != []):
            for c_handle in self.cont_handles:
                c_handle.pop(0).remove()
            self.cont_handles = []
        # Draw new contours, if they exist and flag is true
        if (self.setContours is True):
            contours = self.dv.detect_edges(hp.load_from_stack(self.file_path, self.frame))
            c_extrema = self.dv.find_edge_extrema(image, contours)
            for c in contours:
                c_handle = self.ax.plot(c[:, 1], c[:, 0], lw=2, color='blue')
                self.cont_handles.append(c_handle)
            for c_max in c_extrema:
                c_max_handle = self.ax.plot(c_max[0], c_max[1], 'o', lw=2, color='green')
                self.cont_handles.append(c_max_handle)
        # Show manually selected point, if it exists
        if (not any(np.isnan(self.manual_points[self.frame]))):
            dot_plot = self.ax.plot(self.manual_points[self.frame, 0], self.manual_points[self.frame, 1], 'o', color='red')
            self.cont_handles.append(dot_plot)
        
        # Set image data
        self.im_handle.set_data(image)
        self.ax.set_title(f'frame {self.frame}/{self.length}, start frame {self.start_frame}, manual edge points: {np.sum(~np.isnan(self.manual_points[:, 0]))}')
        self.fig.canvas.draw()
        
    def close(self, event):
        plt.close(self.fig)
        
    def on_close(self, event):
        print('Closing ...')
        print(f'Start frame set at: {self.start_frame}')
        self.exit = True
        
    def get_data(self):
        return self.start_frame, self.manual_points
    
# # Define location measurement suite
# file_path = "D:\\masterproject\\images\\dodecane_17012025\\set2\\meas3\\top_view\\dodecane_top_view.tif"
file_path = "D:\\masterproject\\images\\dodecane_17012025\\set2\\meas3\\side_view\\dodecane_side_view.tif"    

plt.close('all')
iv = ImageVisualizer(file_path, view='top')
