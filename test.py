# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:23:00 2025

@author: coena
"""

import numpy as np
import HelperFunctions as hp
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Define location measurement suite
file_path = "D:\\masterproject\\images\\dodecane_17012025\\set2\\meas1\\dodecane_side_view.tif"


stack_length = hp.get_stack_size(file_path)


class Index:
    ind = 0
    
    def next(self, event):
        self.ind += 1
        i+ self.ind % len()
        
