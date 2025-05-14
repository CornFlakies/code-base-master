# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 09:21:23 2025

@author: Coen Arents
"""

import os
import re
import yaml
import numpy as np
import pandas as pd
import skimage as sk
from tqdm import tqdm
import InitImage as iim
import HelperFunctions as hp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
from ComputeLensDynamics import ComputeLensDynamics

# Load data from Hack and Burton
path_burton = "S:\\masterproject\\scrubbed_data\\burton\\top_view_points.csv"
data_burton = pd.read_csv(path_burton).to_numpy() # in micrometer and microsecs
path_burton = "S:\\masterproject\\scrubbed_data\\burton\\dodecane_01.csv"
data_burton2 = pd.read_csv(path_burton).to_numpy() # centimeters and second
data_burton2[:, 0] *= 1e6
data_burton2[:, 1] *= 1e4
path_burton = "S:\\masterproject\\scrubbed_data\\burton\\dodecane_02.csv"
data_burton3 = pd.read_csv(path_burton).to_numpy() # centimeters and second
data_burton3[:, 0] *= 1e6
data_burton3[:, 1] *= 1e4

path_hack = "S:\\masterproject\\scrubbed_data\\hack\\figure1\\1p36mPa.txt"
data_hack = []
with open(path_hack, "r") as file:
    for line in file:
        data = line.split(" ")
        data_hack.append([float(data[0]), float(data[1])])
data_hack = np.asarray(data_hack)

# Define material paremeters, and relevant characteristic scales
# DODECANE
eta = 1.36e-3 # Pa s
gamma =  25e-3 # N / m
rho = 750 # kg / m^3

# WATER
eta0 = 1e-3 # Pa s
gamma0 = 72e-3 # N / m
rho0 = 998 # kg / m^3

# Different characteristic time- and lengthscales
R = 1e-3 # m
l_v = eta**2 / (gamma * rho) # Capillary length dodecane
t_v = eta**3 / (gamma**2 * rho) # Viscous time scale of dodecane
t_v0 = eta0**3 / (gamma0**2 * rho0) # Viscous time scale of water
tau_v = eta * R / gamma # Viscous length scale of dodecane
t_i = np.sqrt(rho * (R)**3 / gamma) # Inertial time scale of dodecane
Oh = eta / np.sqrt(rho * gamma * R) # Ohnesorge number dodecane
l_c = R * Oh # Crossover length dodecame
t_c = tau_v * Oh # Crossover time dodecane

Rb = 4e-3 # m
l_n = 0.5e-6 # natural viscous length scale
S = 5e-3 # N / m 
Ohb = eta / np.sqrt(rho * S * Rb) # Ohnesorge number with Burtons definitions

# Set mpl font, and fontsizes for both math- and text-mode
import matplotlib 
from cycler import cycler

# A colorset
cmap = plt.cm.Reds

def cross_over(x, C_v=1, C_i=1):
    return (1/(C_v * x) + 1/(C_i * np.sqrt(x)))**(-1)

def cross_over_visc(x, C_v=1):
    return C_v * x

def cross_over_iner(x, C_i=1):
    return C_i * np.sqrt(x)

def burtonius_visc(x):
    return 1.65 / (4 * np.pi) * (S / eta) * x

def burtonius_iner(x):
    return 1.024 * (Rb * S / rho)**(1/4) * x**(1/2)

def find_matching_indices(arr1, arr2):
    '''
    Scanes two 1D arrays and returns the indices where values are equal
    '''
    i, j = 0, 0  # Two pointers
    matching_indices = []

    while i < len(arr1) and j < len(arr2):
        if arr1[i] == arr2[j]:  # Found a match
            matching_indices.append((i, j))
            i += 1
            j += 1
        elif arr1[i] < arr2[j]:  # Move pointer of smaller value
            i += 1
        else:
            j += 1

    return matching_indices

def power_law(height, x_start, x_end, p):
    '''
    Returns the x- and y-values of a power law p.
    '''
    x = np.linspace(x_start, x_end, 2)
    return x, height * (x/x[0])**(p)

def fit_power_law(x, A, p=1/2):
    return A*x**p

def exp_power_law(x, A):
    return A*x**(1/6)

def logtriangle(x_loc, y_loc, baselength, slope, flipped=False):
    '''
    Returns the vertices triangle in loglog space. Needs coordinates for vertex of the
    90 degree angle, a length for the base of the triangle, and a slope.
    Lastly, the flipped parameter will put the orientation at the bottom right if true
    and at the top left if false.
    '''
    x1, y1 = x_loc, y_loc
    x2 = x_loc + baselength
    
    if not flipped:
        y2 = y1 * (x2 / x1) ** slope
        vertices = [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y1],
            ]
    if flipped:
        y2 = y1 * (x1 / x2) ** slope
        vertices = [
            [x1, y2],
            [x1, y1],
            [x2, y1],
            [x1, y2],
            ]
    
    return vertices

def custom_legend(ax, markerscale=2, frameon=False, fontsize=25, loc=None):
    ax.legend(
        markerscale=markerscale,
        frameon=frameon,
        fontsize=fontsize, 
        loc=loc)

#%% Plot data
size = 25

# Create figures
plt.close('all')
fig1, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
ax1 = ax[0]
ax2 = ax[1]
# fig2, ax2 = plt.subplots(figsize=(7, 6))
fig3, ax3 = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
fig4, ax4 = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
fig5, ax5 = plt.subplots(figsize=(7, 6))
fig6, ax6 = plt.subplots(figsize=(7, 6))
fig7, ax7 = plt.subplots(figsize=(7, 6))


# Zoomed side view measurements
abs_path_1 = "S:\\masterproject\\images\\dodecane_17012025\\set2"
file = os.path.join(abs_path_1, 'data.pkl')
df1 = pd.read_pickle(file)

abs_path_2 = "S:\\masterproject\\images\\dodecane_18032025\\set1"
file = os.path.join(abs_path_2, 'data.pkl')
df2 = pd.read_pickle(file)

abs_path_3 = "S:\\masterproject\\images\\dodecane_20032025\\set1"
file = os.path.join(abs_path_3, 'data.pkl')
df3 = pd.read_pickle(file)

abs_path_4 = "S:\\masterproject\\images\\dodecane_26032025\\set2"
file = os.path.join(abs_path_4, 'data.pkl')
df4 = pd.read_pickle(file)

# Loop measurement realizations
r_and_t = []
for ii in df3.index:
    # Load all the data from the dataframe
    r_max_top = df3.loc[ii, 'R_max_top']
    frames_top = df3.loc[ii, 'frames_top']
    R = df3.loc[ii, 'drop_radii']
    fps = df3.loc[ii, 'fps']
    alpha_top = df3.loc[ii, 'alpha_top']
    if (ii == 2):
        continue
    r_and_t.append([r_max_top, frames_top / fps, cmap(.5), 'none'])    

for ii in df4.index:
    # Load all the data from the dataframe
    r_max_top = df4.loc[ii, 'R_max_top']
    frames_top = df4.loc[ii, 'frames_top']
    R = df4.loc[ii, 'drop_radii']
    fps = df4.loc[ii, 'fps']
    alpha_top = df4.loc[ii, 'alpha_top']
    r_and_t.append([r_max_top, frames_top / fps, cmap(.5), 'none'])

for ii in df1.index:    
    # Load all the data from the dataframe
    r_max_top = df1.loc[ii, 'R_max_top']
    frames_top = df1.loc[ii, 'frames_top']
    R = df1.loc[ii, 'drop_radii']
    fps = df1.loc[ii, 'fps']
    alpha_top = df1.loc[ii, 'alpha_top']
    
    r_and_t.append([r_max_top, frames_top / fps, cmap(.8), 'full'])

rdiff = []
first = True
firstSim = True
for (r, t, c1, c2) in r_and_t:
    # r = np.linalg.norm(r - r[0], axis=1)
    r = r[:, 1] - r[0, 1]
    # if not any(np.diff(r)[10:] > 1e-5):
    t = t - t[0]
    # ax1.loglog(t[1:] / t_c, r[1:] / l_c, 'o', color=c)
    if ((c2 =='full') & firstSim):
        ax1.loglog(t[1:] * 1e6, r[1:] * 1e6, 'o', color=c1, fillstyle=c2, label='Simultaneous Data')
        firstSim = False
    elif first:
        ax1.loglog(t[1:] * 1e6, r[1:] * 1e6, 'o', color=c1, fillstyle=c2, label='Extra Data')
        first = False
    else:
        ax1.loglog(t[1:] * 1e6, r[1:] * 1e6, 'o', color=c1, fillstyle=c2)
    
        
    

# popt_visc, _ = curve_fit(power_law_visc, data_burton2[

# Simultaneous measurements
# Open DF
abs_path = "D:\\masterproject\\images\\dodecane_17012025\\set2"
abs_path = "S:\\masterproject\\images\\dodecane_17012025\\set2"
file = os.path.join(abs_path, 'data.pkl')
df = pd.read_pickle(file)

# Normalize the parameters to map them between 0 and 1
R = [df.loc[ii, 'drop_radii'] for ii in df.index]
norm = mcolors.LogNorm(vmin=min(R)*0.7, vmax=max(R)*1.1)

# Select colors for the standard color cycle
hex_colors = np.array(['#a1132f', '#46b3d1', '#77ac31', '#7e2f8c', '#ebb120', '#d95317', '#1a71ad'])

# Set as the default color cycle
matplotlib.rcParams["axes.prop_cycle"] = cycler(color=hex_colors)

# Set font to be latex mathtext
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# Inrease tick- and axis label sizes
matplotlib.rc('xtick', labelsize=size) 
matplotlib.rc('ytick', labelsize=size)
matplotlib.rc('axes', labelsize=size)


# Set unit of the axis labels
unit = 1e6 # Micrometers

# collect all measurements
f_all = []
y0h0 = []
power_laws = []

# Loop measurement realizations
params_top = []
params_side = []
first = True
for ii in df.index:
    # Load all the data from the dataframe
    r_max_top = df.loc[ii, 'R_max_top']
    frames_top = df.loc[ii, 'frames_top']
    r_max_side = df.loc[ii, 'Y_max_side']
    frames_side = df.loc[ii, 'frames_side']
    R = df.loc[ii, 'drop_radii']
    fps = df.loc[ii, 'fps']
    alpha_top = df.loc[ii, 'alpha_top']
    alpha_side = df.loc[ii, 'alpha_side']

    if (ii == 3):
        r_max_side = np.delete(r_max_side, [11, 18], axis=0)
        frames_side = np.delete(frames_side, [11, 18], axis=0)
        

    # Define x-values and y-values of top view plot
    x_top = (frames_top - frames_top[0]) / fps
    r_plot_top = np.linalg.norm(r_max_top - r_max_top[0], axis=1)    
    
    # Define inertial time scale
    t_i = np.sqrt((rho * R**3) / gamma)
    
    # Plot of the top view heights of each individual measurement
    # ax1.loglog(x_top[1:] * unit, r_plot_top[1:] * unit, '.', color=cmap(norm(R)), markersize=8, label=f'{R*1e3:0.3f} mm')#, label=f'R = {R * 1e3:.2f} mm')
    # start=60
    # end=-1
    # popt, pcov = curve_fit(fit_power_law, x_top[start:end], r_plot_top[start:end], p0=[1])
    # power_laws.append([popt[0], R])     
    # x_dat = np.logspace(-4, -2, 50)
    
    if first:
        # Define x- and y-values of side view plot
        x_side = (frames_side - frames_top[0]) / fps
        r_plot_side = np.linalg.norm(r_max_side - r_max_side[0], axis=1)
    
        # Plot of side view heights of each individual measurement
        ax2.loglog(x_side[1:] * unit, r_plot_side[1:] * unit, '.', color=cmap(.8), markersize=8, label='Simultaneous Data')#, label=f'R = {R * 1e3:.2f} mm')
        ax3[1].loglog(x_side[1:] * unit, r_plot_side[1:] * unit, 'o', color=cmap(.8), markersize=8, label='Exp. Data')
        params_side.append(curve_fit(fit_power_law, x_side[1:], r_plot_side[1:], p0=[1, 3/2])[0])
        
        # Plot the top view heights of each individual measurement
        ax3[0].loglog(x_top[1:] * unit, r_plot_top[1:] * unit, 'o', lw=2, color=cmap(.8), label='Exp. Data')
        params_top.append(curve_fit(fit_power_law, x_top[1:], r_plot_top[1:], p0=[1, 1/2])[0])
        
        first = False
    else:
        # Define x- and y-values of side view plot
        x_side = (frames_side - frames_top[0]) / fps
        r_plot_side = np.linalg.norm(r_max_side - r_max_side[0], axis=1)
    
        # Plot of side view heights of each individual measurement
        ax2.loglog(x_side[1:] * unit, r_plot_side[1:] * unit, '.', color=cmap(.8), markersize=8)#, label=f'R = {R * 1e3:.2f} mm')
        ax3[1].loglog(x_side[1:] * unit, r_plot_side[1:] * unit, 'o', color=cmap(.8), markersize=8)
        params_side.append(curve_fit(fit_power_law, x_side[1:], r_plot_side[1:], p0=[1, 3/2])[0])
        
        # Plot the top view heights of each individual measurement
        ax3[0].loglog(x_top[1:] * unit, r_plot_top[1:] * unit, 'o', lw=2, color=cmap(.8))
        params_top.append(curve_fit(fit_power_law, x_top[1:], r_plot_top[1:], p0=[1, 1/2])[0])
    
    # Compute h_0 / y_0
    indices = find_matching_indices(frames_top, frames_side)
    f = []
    theta = []
    for jj in indices[1:]:
        y_0 = r_plot_top[jj[0]]
        h_0 = r_plot_side[jj[1]]
        theta.append(h_0 / y_0)
        f_0 = frames_top[jj[0]]
        f.append(f_0)
        
    t = (f - frames_top[0]) / fps   
    f_all.append(f - frames_top[0])
    y0h0.append(theta)
    ax5.plot(t * 1e3, theta, '.', color=cmap(norm(R)), label=f'{ii}')
    # ax5.plot(*power_law(1.25e-1, 1e-1, 1e0, 1/6), '--', color='black')
    ax5.set_ylabel("$y_0 / h_0$")
    ax5.set_xlabel("$t-t_0 [ms]$")
    ax5.set_ylim([0.1, 0.3])
    # ax5.legend()

    ax6.loglog(x_top[1:] * unit, r_plot_top[1:] * unit, '.', color=cmap(norm(R)), markersize=8, label=f'{R*1e3:0.2f} mm')
    ax6.set_ylim([0.5e2, 1.5e3])


# 
#------------------------------------------------------------------------------
#----------------- Make top view plot with only simultaneous measurements------
#------------------------------------------------------------------------------
ax3[0].set_xlim([0.25e2, 0.5e4])
ax3[0].set_ylim([0.5e2, 1e3])
ax3[1].set_xlim([1e0, 1e4])
ax3[1].set_xlabel('$t - t_0\, [\mu s]$')
ax3[1].set_ylabel('$h_0 [\mu m]$')
ax3[0].set_xlabel('$t - t_0\, [\mu s]$')
ax3[0].set_ylabel('$y_0 [\mu m]$')

x_dat_top = np.logspace(-1, 4, 200)
p_side = np.mean(np.asarray(params_side), axis=0)
x_dat_side = np.logspace(0, 4, 200)
p_top = np.mean(np.asarray(params_top), axis=0)    

ax3[0].loglog(x_dat_top, fit_power_law(x_dat_top * 1e-6, p_top[0], p_top[1]) * 1e6, '-.', lw=3, color='black', label=rf'$y_0\sim {p_top[0]:.3f}\, t^{{{p_top[1]:.3f}}}$')
ax3[1].loglog(x_dat_side, fit_power_law(x_dat_side * 1e-6, p_side[0], p_side[1]) * 1e6, '-.', lw=3, color='black', label=rf'$h_0\sim {p_side[0]:.3f}\, t^{{{p_side[1]:.3f}}}$')
custom_legend(ax3[0])
custom_legend(ax3[1])



#------------------------------------------------------------------------------
#----------------------- MAKE TOP VIEW PLOT WITH BURTON DATA ------------------
#------------------------------------------------------------------------------
# Create fig1 legend so that burton and hack data does not get added
# handles, labels = ax1.get_legend_handles_labels()
# labels, handles = zip(*sorted(zip(labels, handles), key=lambda t:t[0]))
# fig1.legend(handles, labels, loc='upper center', ncols=4, markerscale=2, fontsize=15)

vertices = logtriangle(100, 400, 400, 1/2, flipped=True)

xlabel_loc = [10**((np.log10(vertices[0][0]) + np.log10(vertices[1][0])) / 2),
              10**((np.log10(vertices[0][1]) + np.log10(vertices[1][1])) / 2)]
ylabel_loc = [10**((np.log10(vertices[1][0]) + np.log10(vertices[2][0])) / 2),
              10**((np.log10(vertices[1][1]) + np.log10(vertices[2][1])) / 2)]

# Unzip the vertices into x and y coordinates
x, y = zip(*vertices)

# Define parameters from Burton paper
C_i = 1.024
C_v = 1.65 / 4 / np.pi
l_cb = 253e-6 # from paper
t_cb = 700e-6

# t_cb = (2 * np.pi * eta)**2 * np.sqrt(Rb / (S**3 * rho)) # equating velocities
x_ana = np.logspace(-20, 5, 200)
visc = burtonius_visc(x_ana)
iner = burtonius_iner(x_ana) 
ax6.loglog(x_ana * 1e6, visc * 1e6, '-.', color='black', label=r'$1.65\, (St/4\pi\eta_0) $')
ax6.loglog(x_ana * 1e6, iner * 1e6, '--', color='black', label=r'$1.024\, (R S / \rho_0)^{1/4}\, t^{1/2} $')
ax6.loglog(x_ana * 1e6, cross_over(x_ana / t_cb, C_v, C_i) * l_cb * 1e6, '-', color='black', label=r'$f(\xi)\, l_c$')
ax6.plot(x, y, color='black')
ax6.annotate('2', xlabel_loc, textcoords="offset points", xytext=(-15, 0), fontsize=25, ha='right')
ax6.annotate('1', ylabel_loc, textcoords="offset points", xytext=(0, 15), fontsize=25, ha='center')
ax6.set_xlabel(r'$t - t_0 [\mu s]$')
ax6.set_ylabel('$y_0 [\mu m]$')
ax6.loglog(data_burton[:, 0], data_burton[:, 1], 'd', color='black', label='Burton (scrubbed)')
ax6.loglog(data_burton2[:, 0], data_burton2[:, 1], 'd', color='green', label='Burton (original)')
ax6.loglog(data_burton3[:, 0], data_burton3[:, 1], 'd', color='green')

x_ana = np.linspace(1e-7, 1e0, 80)
ax6.legend([ax6.lines[-1]], ['Burton & Taborek 2007'], 
           markerscale=2,
           frameon=False,
           fontsize=15, 
           loc='upper left')
ax6.set_xlim([1e1, 1e4])

# Plot triangle and add side labels
ax1.plot(x, y, color='black')
ax1.annotate('2', xlabel_loc, textcoords="offset points", xytext=(-15, 0), fontsize=25, ha='right')
ax1.annotate('1', ylabel_loc, textcoords="offset points", xytext=(0, 15), fontsize=25, ha='center')
ax1.loglog(data_burton[:, 0], data_burton[:, 1], 'd', color='black', label='Burton &\nTaborek. 2007')
# ax1.legend([ax1.lines[-1]], ['Burton & Taborek 2007'], 
#            markerscale=2,
#            frameon=False,
#            fontsize=15, 
#            loc='upper left')
custom_legend(ax1, fontsize=12.5, loc='upper left')
# custom_legend(ax1, fontsize=18, loc='upper left')
# ax1.set_title("Top view, $y_0$ as a function of time")
ax1.set_ylim([1e1, 1.5e3])
ax1.set_xlabel('$t - t_0\, [\mu s]$')
ax1.set_ylabel('$y_0 [\mu m]$')
ax1.xaxis.set_ticks_position('both')
ax1.yaxis.set_ticks_position('both')

#------------------------------------------------------------------------------
#------------------------ MAKE SIDE VIEW PLOT WITH HACK DATA ------------------
#------------------------------------------------------------------------------
# Vertices from log triangle
vertices = logtriangle(210, 20, 1000, 2/3, flipped=False)

xlabel_loc = [10**((np.log10(vertices[0][0]) + np.log10(vertices[1][0])) / 2),
              10**((np.log10(vertices[0][1]) + np.log10(vertices[1][1])) / 2)]
ylabel_loc = [10**((np.log10(vertices[1][0]) + np.log10(vertices[2][0])) / 2),
              10**((np.log10(vertices[1][1]) + np.log10(vertices[2][1])) / 2)]

# Unzip the vertices into x and y coordinates
x, y = zip(*vertices)

# Plot triangle and add side labels
ax2.plot(x, y, color='black')
ax2.annotate('3', xlabel_loc, textcoords="offset points", xytext=(0, -20), fontsize=25, ha='center')
ax2.annotate('2', ylabel_loc, textcoords="offset points", xytext=(20, 0), fontsize=25, ha='right')

# Plot the data from Hack
ax2.loglog(data_hack[:, 0] * 1e6, data_hack[:, 1] * 1e6, '>', color='black', label='Hack et al. 2020')
ax2.set_xlim([0.5e1, 1e4])

# Set custom legend
# ax2.legend([ax2.lines[-1]], ['Hack et al. 2020'], 
#            markerscale=2,
#            frameon=False,
#            fontsize=15, 
#            loc='upper left')
custom_legend(ax2, fontsize=15)
# custom_legend(ax2, fontsize=20, loc='upper left')
ax2.set_ylabel('$h_0 [\mu m]$')
ax2.set_xlabel('$t - t_0\, [\mu s]$')
# ax2.set_title("Side view, $h_0$ as a function of time")
ax2.xaxis.set_ticks_position('both')
ax2.yaxis.set_ticks_position('both')

# ax3.set_title("top view, divided by the drop radius")
# xi = np.logspace(-2, 7, 100)
# ax3.loglog(data_burton[:, 0] * 1e-6 / t_c, data_burton[:, 1] * 1e-6 / l_c, '>', color='black', label='Burton & Taborek 2007')
# ax3.loglog(xi, cross_over(xi), '-', color='black', label=r'$f\, (\xi)$')
# ax3.loglog(xi, cross_over_visc(xi), '--', color='black', label=r'$C_v \xi$')
# ax3.loglog(xi, cross_over_iner(xi), '-.', color='black', label=r'$C_i \sqrt{\xi}$')
# ax3.set_ylim([1e-2, 1e3])
# ax3.set_xlim([1e-1, 1e7])
# ax3.set_xlabel(r'$t/t_c$')
# ax3.set_ylabel('$y_0/l_c$')
# custom_legend(ax3, fontsize=15, markerscale=1.5, loc='lower right')
# ax3.minorticks_on()
# ax3.xaxis.set_ticks_position('both')
# ax3.yaxis.set_ticks_position('both')


# ax3.set_title("top view, divided by the drop radius")
ax6.set_xlabel(r'$t - t_0\, [\mu s]$')
ax6.set_ylabel('$y_0\, [\mu m]$')

#------------------------------------------------------------------------------
# --------------------------------- MAKE Y0H0 PLOT ----------------------------
#------------------------------------------------------------------------------
fmax = 0
for f in f_all:
    if (f[-1] > fmax):
        fmax = f[-1]
        
data = np.empty((len(f_all), fmax))
data[:] = np.nan
for ii, (f, yh) in enumerate(zip(f_all, y0h0)):
    data[ii, f-1] = yh 
mean = np.nanmean(data, axis=0)
std = np.nanstd(data, axis=0)
t = np.arange(0, len(mean)) / fps

# Set start and cutoff params
start=5
cutoff = -40
t *= 1e3

# Plot the mean of y0/h0
ax4[1].errorbar(t[start:cutoff], mean[start:cutoff], yerr=std[start:cutoff], fmt='+', color='green')
ax4[0].set_xlabel(r'$(t - t_0)[ms]$')
ax4[0].set_ylabel('$y_0 / h_0$')
ax4[1].set_ylim([0.15, 0.26])

ax4[0].errorbar(t[start:cutoff], mean[start:cutoff], yerr=std[start:cutoff], fmt='+', color='green')
# custom_legend(ax4[0])
ax4[1].set_xlabel(r'$(t - t_0)[ms]$')


# Fit power law to data
idx = ~np.isnan(mean)
popt, pcov = curve_fit(fit_power_law, t[idx], mean[idx], p0=[1, 1/6])
x_dat = np.linspace(t[start], t[cutoff], 200)
ax4[0].loglog(x_dat, fit_power_law(x_dat, *popt), '-.', lw=3, color='black', label=rf'$y_0/h_0 = {popt[0]:.3f}\, t^{{{popt[1]:.3f}}}$')
ax4[1].plot(x_dat, fit_power_law(x_dat, *popt), '-.', lw=3, color='black', label=rf'$y_0/h_0 = {popt[0]:.3f}\, t^{{{popt[1]:.3f}}}$')

# Fit power law to data
idx = ~np.isnan(mean)
popt, pcov = curve_fit(exp_power_law, t[idx], mean[idx], p0=[1])
x_dat = np.linspace(t[start], t[cutoff], 200)
ax4[0].loglog(x_dat, exp_power_law(x_dat, *popt), '--', lw=3, color='black', label=rf'$y_0/h_0 = {popt[0]:.3f}\,$' + r'$t^{1/6}$')
ax4[1].plot(x_dat, exp_power_law(x_dat, *popt), '--', lw=3, color='black', label=rf'$y_0/h_0 = {popt[0]:.3f}\,$' + r'$t^{1/6}$')

custom_legend(ax4[1])

# Create a secondary y-axis with numeric values, but keep it aligned
ax8 = ax4[1].twinx()
# ax8.set_yticks(np.linspace(0.14, 0.24, 10))
y_ticks = np.linspace(0.15, 0.26, 10) * 180 / np.pi
ax8.set_yticklabels(
    [f"{label:.1f}" for label in y_ticks]
);
ax8.set_ylabel(r"$\theta$")

#------------------------------------------------------------------------------
# --------------------------------- MAKE A(R) plot ----------------------------
#------------------------------------------------------------------------------
Adat = []
Rdat = []
for A, R in power_laws:
    Adat.append(A)
    Rdat.append(R)
idx = np.argsort(Rdat)
Rdat = np.asarray(Rdat)[idx]
Adat = np.asarray(Adat)[idx]
ax7.plot(Rdat, Adat, '-o')

# Make tight layouts
fig1.tight_layout()
fig1.subplots_adjust(top=0.85)
# fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig5.tight_layout()
fig6.tight_layout()
fig7.tight_layout()

plt.show()

#%% Make plot with zoomed out data of top view
plt.close('all')

abs_path_1 = "S:\\masterproject\\images\\dodecane_17012025\\set2"
file = os.path.join(abs_path_1, 'data.pkl')
df1 = pd.read_pickle(file)

abs_path_2 = "S:\\masterproject\\images\\dodecane_18032025\\set1"
file = os.path.join(abs_path_2, 'data.pkl')
df2 = pd.read_pickle(file)

abs_path_3 = "S:\\masterproject\\images\\dodecane_20032025\\set1"
file = os.path.join(abs_path_3, 'data.pkl')
df3 = pd.read_pickle(file)


r_and_t = []

# Loop measurement realizations
for ii in df1.index:    
    # Load all the data from the dataframe
    r_max_top = df1.loc[ii, 'R_max_top']
    frames_top = df1.loc[ii, 'frames_top']
    R = df1.loc[ii, 'drop_radii']
    fps = df1.loc[ii, 'fps']
    alpha_top = df1.loc[ii, 'alpha_top']
    
    r_and_t.append([r_max_top, frames_top / fps, 'red'])
    
for ii in df2.index:
    # Load all the data from the dataframe
    r_max_top = df2.loc[ii, 'R_max_top']
    frames_top = df2.loc[ii, 'frames_top']
    R = df2.loc[ii, 'drop_radii']
    fps = df2.loc[ii, 'fps']
    alpha_top = df2.loc[ii, 'alpha_top']
    r_and_t.append([r_max_top, frames_top / fps, 'blue'])
    
for ii in df3.index:
    # Load all the data from the dataframe
    r_max_top = df3.loc[ii, 'R_max_top']
    frames_top = df3.loc[ii, 'frames_top']
    R = df3.loc[ii, 'drop_radii']
    fps = df3.loc[ii, 'fps']
    alpha_top = df3.loc[ii, 'alpha_top']
    if (ii == 2):
        continue
    r_and_t.append([r_max_top, frames_top / fps, 'green'])
    
fig1, ax1 = plt.subplots()

rdiff = []
for (r, t, c) in r_and_t:
    # r = np.linalg.norm(r - r[0], axis=1)
    r = r[:, 1] - r[0, 1]
    # if not any(np.diff(r)[10:] > 1e-5):
    t = t - t[0]
    ax1.loglog(t[1:] * 1e6, r[1:] * 1e6, 'o', color=c)
        # ax1.loglog(np.diff(r)[10:])
        
# Create fig1 legend so that burton and hack data does not get added
vertices = logtriangle(100, 400, 400, 1/2, flipped=True)

xlabel_loc = [10**((np.log10(vertices[0][0]) + np.log10(vertices[1][0])) / 2),
              10**((np.log10(vertices[0][1]) + np.log10(vertices[1][1])) / 2)]
ylabel_loc = [10**((np.log10(vertices[1][0]) + np.log10(vertices[2][0])) / 2),
              10**((np.log10(vertices[1][1]) + np.log10(vertices[2][1])) / 2)]

# Unzip the vertices into x and y coordinates
x, y = zip(*vertices)
ax1.plot(x, y, color='black')
ax1.annotate('2', xlabel_loc, textcoords="offset points", xytext=(-15, 0), fontsize=25, ha='right')
ax1.annotate('1', ylabel_loc, textcoords="offset points", xytext=(0, 15), fontsize=25, ha='center')

vertices = logtriangle(800, 200, 2000, 1/2, flipped=False)
# Unzip the vertices into x and y coordinates
x, y = zip(*vertices)
ax1.plot(x, y, color='black')

ax1.axhline(y=1000, linestyle='--', color='black', lw=2, label=r'$R_drop$')

ax1.set_xlabel(r'$t - t_0 [\mu s]$')
ax1.set_ylabel('$y_0 [\mu m]$')

fig1.tight_layout()
