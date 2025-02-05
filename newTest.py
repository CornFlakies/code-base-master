# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:15:03 2025

@author: coena
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np


class ImageVisualizer:

    def __init__(self):
        # Test image stack (simulated)
        self.images = [np.random.random((10, 10)) for _ in range(5)]
        self.ind = 0

        # Setup figure
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)  # Add space for buttons

        # Initial image
        self.im_handle = self.ax.imshow(self.images[self.ind], cmap='gray')

        # Define buttons
        axprev = self.fig.add_axes([0.7, 0.05, 0.1, 0.075])
        axnext = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])

        # Create buttons and bind events
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(self.next)

        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(self.prev)

        plt.show()

    def next(self, event):
        self.ind = (self.ind + 1) % len(self.images)
        self.update_image()

    def prev(self, event):
        self.ind = (self.ind - 1) % len(self.images)
        self.update_image()

    def update_image(self):
        self.im_handle.set_data(self.images[self.ind])
        self.fig.canvas.draw()


# Instantiate and run
ImageVisualizer()
