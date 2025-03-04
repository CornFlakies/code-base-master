# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:35:01 2024

@author: coena
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

plt.close('all')


# Load data from Hack and Burton
path_burton = "D:\\masterproject\\scrubbed_data\\burton\\top_view_points.csv"
data_burton = pd.read_csv(path_burton).to_numpy() # in micrometer and microsecs

path_hack = "D:\\masterproject\\scrubbed_data\\hack\\figure1\\1p36mPa.txt"
data_hack = []
with open(path_hack) as file:    
    for line in file:
        data_hack.append([float(x)  for x in line.split()])
data_hack = np.asarray(data_hack, dtype=float)
    
# Define location measurement suite
abs_path = "D:\\masterproject\\images\\dodecane_17012025\\set2"
# abs_path = "S:\\masterproject\\images\\dodecane_17012025\\set2"

eta = 1.36e-3 # Pa s
gamma =  25e-3 # N / m
rho = 750 # kg / m^3
# l_v = eta**2 / (gamma * rho)

#%% 

# If the config files do not exist, create new ones.
isLive = False
if isLive:
    # FPS needs to be defined by the user
    fps = 1e5
    iim.create_config_from_suite(abs_path, fps)
    
# Load the files and directories from path
contents = os.listdir(abs_path)

# Create list to prepare data for pandas dataframe
df_data = []

# Create lists for the r data of each measurement
R_all_side = []
R_all_top = []
all_frames_side = []
all_frames_top = []
all_radii = []

# For each measurement, compute the initial radius of the droplets
for directory in contents:
    if bool(re.search("meas", directory)):
        with open(os.path.join(abs_path, directory,'config.yaml')) as file:
            try:
                data = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
            
            # Pre-load data from the manually selected points
            r_max_side = np.load(os.path.join(abs_path, directory, 'manual_points_side.npy'))
            r_max_top = np.load(os.path.join(abs_path, directory, 'manual_points_top.npy'))
            
            # Load the rest of the measurement data
            frame_start_top = data['INITIAL_PARAMETERS']['INITIAL_FRAME_TOP_VIEW']
            frame_start_side = data['INITIAL_PARAMETERS']['INITIAL_FRAME_SIDE_VIEW']
            R = data['INITIAL_PARAMETERS']['DROP_RADIUS']
            alpha_top = data['MEASUREMENTS_PARAMETERS']['CONV_TOP_VIEW']
            alpha_side = data['MEASUREMENTS_PARAMETERS']['CONV_SIDE_VIEW']
            fps = data['MEASUREMENTS_PARAMETERS']['FPS']
            
            # Get drop radii
            all_radii.append(R)
            
            # Compute heights
            input_dir = os.path.join(abs_path, directory, 'side_view')
            cd = ComputeLensDynamics(input_dir, 
                                      XMIN=0, XMAX=None, 
                                      YMIN=0, YMAX=None, 
                                      framestart=('stack 1', frame_start_side), 
                                      view='side')
            data_side = cd.get_R()
            
            # add the manually selected data to the computed data, then append
            # to a list containing all the individual measurement data
            for ii, r in enumerate(data_side): # Side view
                if (r != []):
                    r_max_side[frame_start_side + ii, :] = [r[0][0], r[0][1]]
            frames_side = np.where(~np.isnan(r_max_side[:, 0]))[0]
            all_frames_side.append(frames_side)
            r_max_side = r_max_side[~np.isnan(r_max_side).any(axis=1)]
            R_all_side.append(r_max_side)
        
            # Compute heights
            input_dir = os.path.join(abs_path, directory, 'top_view')
            cd = ComputeLensDynamics(input_dir, 
                                      XMIN=0, XMAX=None, 
                                      YMIN=0, YMAX=None, 
                                      framestart=('stack 1', frame_start_top), 
                                      view='top')
            data_top = cd.get_R()
            
            for ii, r in enumerate(data_top): # Top view
                if (r != []):
                    r_max_top[frame_start_top + ii, :] = [r[0][0], r[0][1]]
                    
            # Get indices of all non nan elements
            frames_top = np.where(~np.isnan(r_max_top[:, 0]))[0]
            all_frames_top.append(frames_top)
            r_max_top = r_max_top[~np.isnan(r_max_top).any(axis=1)] 
            R_all_top.append(r_max_top)
            
            # Define data
            data = {"R_max_top": r_max_top * alpha_top,
                    "frames_top": frames_top,
                    "Y_max_side": r_max_side * alpha_side,
                    "frames_side": frames_side,
                    "drop_radii": R,
                    "fps" : fps,
                    "alpha_top": alpha_top,
                    "alpha_side": alpha_side}
            
            # append to list
            df_data.append(data)
            
# Create data frame from dicts
df = pd.DataFrame(df_data) 

# Save to csv file
file = os.path.join(abs_path, 'data.pkl')
df.to_pickle(file)

#%% Replace R

# Load the files and directories from path
contents = os.listdir(abs_path)

# Open DF
file = os.path.join(abs_path, 'data.pkl')
df = pd.read_pickle(file)
ii = 0

# Loop measurement realizations
for directory in contents:
    if bool(re.search("meas", directory)):
        print(directory)
        
        # Load init view images and get drop radii
        path_init_view = os.path.join(abs_path, directory, 'init_top_view')
        path_init_view_img, init_view_img = hp.load_files(path_init_view, header="tif")
        init_view_img = sk.io.imread(path_init_view_img[0])
        
        # Find radii
        radii, centers = iim.get_drop_radii(init_view_img)
        alpha_top = df.loc[ii, 'alpha_top']
        
        # Convert radii to effective curvature, then convert to meters
        R = (radii[0] * radii[1]) / (radii[0] + radii[1])
        R *= alpha_top
        
        # Save R
        df.loc[ii, 'drop_radii'] = R
        
        # truncate index
        ii += 1
        
# Save to csv file
file = os.path.join(abs_path, 'data.pkl')
df.to_pickle(file)

#%% Plot data
size = 25

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
    
# Open DF
file = os.path.join(abs_path, 'data.pkl')
df = pd.read_pickle(file)

# Set mpl font, and fontsizes for both math- and text-mode
import matplotlib 
from cycler import cycler

# A colorset
cmap = plt.cm.Reds

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

# Create figures
plt.close('all')
fig1, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
ax1 = ax[0]
ax2 = ax[1]
# fig2, ax2 = plt.subplots(figsize=(7, 6))
fig3, ax3 = plt.subplots(figsize=(7, 6))
fig4, ax4 = plt.subplots(figsize=(7, 6))
fig5, ax5 = plt.subplots(figsize=(7, 6))
fig6, ax6 = plt.subplots(figsize=(7, 6))
fig7, ax7 = plt.subplots(figsize=(7, 6))

# Set unit of the axis labels
unit = 1e6 # Micrometers

# collect all measurements
f_all = []
y0h0 = []
power_laws = []

# Loop measurement realizations
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
        

    # Define x-values
    x_top = (frames_top - frames_top[0]) / fps
    r_plot_top = np.linalg.norm(r_max_top - r_max_top[0], axis=1)    
    
    # Define inertial time scale
    t_i = np.sqrt((rho * R**3) / gamma)
    
    # Plot of the top view heights of each individual measurement
    ax1.loglog(x_top[1:] * unit, r_plot_top[1:] * unit, '.', color=cmap(norm(R)), markersize=8, label=f'{R*1e3:0.3f} mm')#, label=f'R = {R * 1e3:.2f} mm')
    start=60
    end=-1
    popt, pcov = curve_fit(fit_power_law, x_top[start:end], r_plot_top[start:end], p0=[1])
    power_laws.append([popt[0], R])     
    x_dat = np.logspace(-4, -2, 50)
    ax3.loglog(x_dat, fit_power_law(x_dat, *popt), '.-')

    x_side = (frames_side - frames_top[0]) / fps
    r_plot_side = np.linalg.norm(r_max_side - r_max_side[0], axis=1)

    # Plot of side view heights of each individual measurement
    ax2.loglog(x_side[1:] * unit, r_plot_side[1:] * unit, '.', color=cmap(norm(R)), markersize=8)#, label=f'R = {R * 1e3:.2f} mm')
    
    # Plot the top view heights of each individual measurement, divided over
    # the intrinsic length scale
    ax3.loglog(x_top[1:], r_plot_top[1:], '.', lw=2, color=cmap(norm(R)))
    
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
    ax5.plot(t * 1e3, theta, '.', label=f'{ii}')
    # ax5.plot(*power_law(1.25e-1, 1e-1, 1e0, 1/6), '--', color='black')
    ax5.set_ylabel("$y_0 / h_0$")
    ax5.set_xlabel("$t-t_0 [ms]$")
    ax5.set_ylim([0.1, 0.3])
    # ax5.legend()
    
    ax6.loglog(x_top[1:] * unit, r_plot_top[1:] * unit, '.', color=cmap(norm(R)), markersize=8, label=f'{R*1e3:0.2f} mm')
    ax6.set_ylim([0.5e2, 1.5e3])

#------------------------------------------------------------------------------
#----------------------- MAKE TOP VIEW PLOT WITH BURTON DATA ------------------
#------------------------------------------------------------------------------
# Create fig1 legend so that burton and hack data does not get added
handles, labels = ax1.get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t:t[0]))
fig1.legend(handles, labels, loc='upper center', ncols=4, markerscale=2, fontsize=15)

vertices = logtriangle(100, 400, 400, 1/2, flipped=True)

xlabel_loc = [10**((np.log10(vertices[0][0]) + np.log10(vertices[1][0])) / 2),
              10**((np.log10(vertices[0][1]) + np.log10(vertices[1][1])) / 2)]
ylabel_loc = [10**((np.log10(vertices[1][0]) + np.log10(vertices[2][0])) / 2),
              10**((np.log10(vertices[1][1]) + np.log10(vertices[2][1])) / 2)]

# Unzip the vertices into x and y coordinates
x, y = zip(*vertices)

ax6.plot(x, y, color='black')
ax6.annotate('2', xlabel_loc, textcoords="offset points", xytext=(-15, 0), fontsize=25, ha='right')
ax6.annotate('1', ylabel_loc, textcoords="offset points", xytext=(0, 15), fontsize=25, ha='center')
ax6.set_xlabel(r'$t - t_0 [\mu s]$')
ax6.set_ylabel('$y_0 [\mu m]$')

# Plot triangle and add side labels
ax1.plot(x, y, color='black')
ax1.annotate('2', xlabel_loc, textcoords="offset points", xytext=(-15, 0), fontsize=25, ha='right')
ax1.annotate('1', ylabel_loc, textcoords="offset points", xytext=(0, 15), fontsize=25, ha='center')
ax1.loglog(data_burton[:, 0], data_burton[:, 1], 'd', color='black', label='Burton &\nTaborek. 2007')
ax1.legend([ax1.lines[-1]], ['Burton & Taborek 2007'], 
           markerscale=2,
           frameon=False,
           fontsize=15, 
           loc='upper left')
# custom_legend(ax1, fontsize=18, loc='upper left')
# ax1.set_title("Top view, $y_0$ as a function of time")
ax1.set_ylim([0.5e2, 1.5e3])
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
ax2.legend([ax2.lines[-1]], ['Hack et al. 2017'], 
           markerscale=2,
           frameon=False,
           fontsize=15, 
           loc='upper left')
# custom_legend(ax2, fontsize=20, loc='upper left')
ax2.set_ylabel('$h_0 [\mu m]$')
ax2.set_xlabel('$t - t_0\, [\mu s]$')
# ax2.set_title("Side view, $h_0$ as a function of time")
ax2.xaxis.set_ticks_position('both')
ax2.yaxis.set_ticks_position('both')

# ax3.set_title("top view, divided by the drop radius")
ax3.set_xlabel(r'$(t - t_0) / t_i$')
ax3.set_ylabel('$y_0/R$')

# ax3.set_title("top view, divided by the drop radius")
ax6.set_xlabel(r'$t - t_0\, [\mu s]$')
ax6.set_ylabel('$y_0\, [\mu s]$')

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

# Fit power law to data
idx = ~np.isnan(mean)
popt, pcov = curve_fit(fit_power_law, t[idx], mean[idx], p0=[1, 1/6])
x_dat = np.linspace(t[start], t[cutoff], 200)

# Plot the mean of y0/h0
ax4.errorbar(t[start:cutoff], mean[start:cutoff], yerr=std[start:cutoff], fmt='+', color='green')
ax4.plot(x_dat, fit_power_law(x_dat, *popt), '-.', lw=3, color='black', label=rf'$y_0/h_0 \sim t^{{{popt[1]:.3f}}}$')
custom_legend(ax4)
ax4.set_xlabel(r'$(t - t_0)[ms]$')
ax4.set_ylabel('$y_0 / h_0$')

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
