# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:50:06 2025

@author: coena
"""

import os
import re
import yaml
import numpy as np
import pandas as pd
import skimage as sk
from tqdm import tqdm
# import InitImage as iim
import HelperFunctions as hp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
from ComputeLensDynamics import ComputeLensDynamics

path_burton = "S:\\masterproject\\scrubbed_data\\burton\\dodecane_01.csv"
data_burton2 = pd.read_csv(path_burton).to_numpy() # centimeters and second
data_burton2[:, 0] *= 1
data_burton2[:, 1] *= 1e-2
path_burton = "S:\\masterproject\\scrubbed_data\\burton\\dodecane_02.csv"
data_burton3 = pd.read_csv(path_burton).to_numpy() # centimeters and second
data_burton3[:, 0] *= 1
data_burton3[:, 1] *= 1e-2

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

# Definition material parameters
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

Rb = 1e-3 # m
l_n = 0.5e-6 # natural viscous length scale
S = 5e-3 # N / m 
Ohb = eta / np.sqrt(rho * S * Rb) # Ohnesorge number with Burtons definitions
tau_vb = eta * R / S # viscous length scale with spreading coefficient

lim_visc = 80
lim_iner = 40

plt.close('all')

def power_law_iner(x, A):
    return A * np.sqrt(x)

def power_law_visc(x, A):
    return A * x

def cross_over(x, C_v=1, C_i=1):
    return (1/(C_v * x) + 1/(C_i * np.sqrt(x)))**(-1)

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
    
for ii in df3.index:
    # Load all the data from the dataframe
    r_max_top = df3.loc[ii, 'R_max_top']
    frames_top = df3.loc[ii, 'frames_top']
    R = df3.loc[ii, 'drop_radii']
    fps = df3.loc[ii, 'fps']
    alpha_top = df3.loc[ii, 'alpha_top']
    if (ii == 2):
        continue
    r_and_t.append([r_max_top, frames_top / fps, 'blue'])
    
for ii in df4.index:
    # Load all the data from the dataframe
    r_max_top = df4.loc[ii, 'R_max_top']
    frames_top = df4.loc[ii, 'frames_top']
    R = df4.loc[ii, 'drop_radii']
    fps = df4.loc[ii, 'fps']
    alpha_top = df4.loc[ii, 'alpha_top']
    r_and_t.append([r_max_top, frames_top / fps, 'green'])

# Define crossover time and length to create new fits
t_c = tau_vb * Ohb
l_c = 1e-1 * Ohb

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
rdiff = []
for (r, t, c) in r_and_t:
    # r = np.linalg.norm(r - r[0], axis=1)
    r = r[:, 1] - r[0, 1]
    # if not any(np.diff(r)[10:] > 1e-5):
    t = t - t[0]
    ax1.loglog(t[1:] / t_c, r[1:] / l_c, 'o', color=c)
    ax2.loglog(t[1:] * 1e6, r[1:] * 1e6, 'o', color=c)

popt_visc, _ = curve_fit(power_law_visc, data_burton2[lim_visc:, 0] / t_c, data_burton2[lim_visc:, 1] / l_c, p0=[1.65/(4*np.pi)])
popt_iner, _ = curve_fit(power_law_iner, data_burton3[:lim_iner, 0] / t_c, data_burton3[:lim_iner, 1] / l_c, p0=[1.024])
C_v = popt_visc[0]
C_i = popt_iner[0]

# Define crossover time and length
t_c = tau_vb * Ohb
l_c = 1e-1 * Ohb 

ax1.loglog(data_burton2[:, 0] / t_c, data_burton2[:, 1] / l_c, 'd', color='black', label='Burton (original)')
ax1.loglog(data_burton3[:, 0] / t_c, data_burton3[:, 1] / l_c, 'd', color='black')
x_ana = np.logspace(0, 4, 50)
ax1.loglog(x_ana, power_law_visc(x_ana, *popt_visc), '--', color='black', label=r'$r = C_v t$')
ax1.loglog(x_ana, power_law_iner(x_ana, *popt_iner), '--', color='black', label=r'$r = C_i t^{1/2}$')
ax1.loglog(x_ana, cross_over(x_ana , popt_visc[0], popt_iner[0]), color='red', label='crossover-func')
ax1.set_xlabel(r'$t / t_c$')
ax1.set_ylabel(r'$r / r_c$')
ax1.legend()
fig1.tight_layout()

ax2.loglog(data_burton2[:, 0] * 1e6, data_burton2[:, 1] * 1e6, 'd', color='black', label='Burton (original)')
ax2.loglog(data_burton3[:, 0] * 1e6, data_burton3[:, 1] * 1e6, 'd', color='black')
ax2.axhline(y=1000, linestyle='--', color='black', lw=2, label=r'$R_drop$')
ax2.set_xlabel(r'$t - t_0 [\mu s]$')
ax2.set_ylabel('$y_0 [\mu m]$')
fig2.tight_layout()

print(popt_visc[0])
print(popt_iner[0])
# C_v = 1.65 / 4 / np.pi 
# C_i = 1.024

plt.figure()
plt.loglog(data_burton2[:, 0] * 1e6, data_burton2[:, 1] * 1e6, 'd', color='green', label='Burton (original)')
plt.loglog(data_burton3[:, 0] * 1e6, data_burton3[:, 1] * 1e6, 'd', color='green')
x_ana = np.logspace(-5, -1, 50)
plt.loglog(x_ana * 1e6, power_law_visc(x_ana, C_v * S / eta) * 1e6, '-', color='black', label=r'$0.126\, S\tau / \eta$')
plt.loglog(x_ana * 1e6, power_law_iner(x_ana, C_i * (R * S / rho)**(1/4)) * 1e6, '-.', color='black', label=r'$1.422\, (R S / \rho)^{1/4}\, \tau^{1/2}$')
plt.loglog(x_ana * 1e6, cross_over(x_ana, C_v * S / eta, C_i * (R * S / rho)**(1/4)) * 1e6, '-', color='red', label='crossover-func')
plt.xlabel(r'Time [$\mu s$]')
plt.ylabel(r'Neck Radius [$\mu m$]')
plt.legend()
plt.tight_layout()


