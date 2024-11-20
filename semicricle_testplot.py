# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 20:10:22 2024

@author: Coen Arents
"""
import numpy as np
import matplotlib.pyplot as plt

r = 1
x0 = 0
y0 = 0
x_ana = np.linspace(-r, r, 100)

def semicircle(x, x0, y0, r):
    return y0 + np.sqrt(r - (x - x0)**2)

def semicircle(x, x0, y0, r):
    return y0 - np.sqrt(r - (x - x0)**2)

plt.figure()
plt.plot(x_ana, semicircle(x_ana, x0, y0, r))

