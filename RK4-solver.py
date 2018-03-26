'''
Script solves Boltzmann equation for pseudopotential z(barX) in S. Weinberg's model 
for two Goldstone bosons annihilationg into muon/pion pairs

Method: Runge-Kutta, 4th order
Note: Mathematica 10 scripts calculate all relevant cross sections

Approximations:
- pure MB
- fractional BE (fFD)
- partial BE (pBE)
- full BE/FD

Arguments:
- scan: scan name
- ACC: accuracy (important in Mathematica scripts)
- lam_phi: self-coupling constant for phi field
- kappa: coupling constant between phi and Higgs fields
- m1: first scalar's mass in GeV
- calc_interp: whether or not to perform interpolation for MB and pBE approximations
- GG: (optional) whether to include backraction in g(x) function
'''

import os
import sys
import timeit
import numpy as np
import pandas as pd
import math as m
import matplotlib.pyplot as plt
import scipy.special as special
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from utils import join_line, file_append

start = timeit.default_timer()

# Arguments

scan = sys.argv[1]
ACC = int(sys.argv[2])
lam_phi = float(sys.argv[3])
kappa = float(sys.argv[4])
m1 = float(sys.argv[5])
calc_interp = int(sys.argv[6])
GG = sys.argv[7]

# Parameters

Mpl = 1.2209e19 # Planck mass in GeV
mF = 0.1056  # muon mass in GeV
m2 = 125 # second scalar mass (Higgs boson) in GeV
Gam1 = lam_phi*m1/16/m.pi*1000 # first scalar decay rate in GeV
Gam2 = 4 # second scalar deacy rate in GeV

zeta_3 = 1.20205690315959429 # Riemann zeta function for arg = 3
Cz = m.sqrt(45)/128/pow(m.pi,4.5)*Mpl/mF
Cz_MB = m.sqrt(45)/512/pow(m.pi,4.5)*Mpl/mF
CY = 1/m.sqrt(45*m.pi)/512*Mpl/mF
CY_pBE = 1/m.sqrt(45*m.pi)/512*Mpl/mF
coeff = 2*kappa**2*mF**8/m1**4/m2**4 # coefficient in |M|^2

barX_max = 4 # mF/T_max, starting moment for RK4 method

# Paths

path_results = "./results/" + scan + "_" + str(lam_phi) + "/"
path_scripts = "./scripts/"
path_data = "./data/"

#==========================================================================
# Read h(T), g(T) and gstar(T) functions from data folder (see 1503.03513)
#==========================================================================

heff = pd.read_csv(path_data + "heffdhs.dat", sep="\t", header=None)[1] # take only the second column
geff = pd.read_csv(path_data + "geffdhs.dat", sep="\t", header=None)[1]
df_gstar = pd.read_csv(path_data + "gstardhs.dat", sep="\t", names = ['T', 'gstar'])

temps = df_gstar['T'].values
gstar = df_gstar['gstar'].values

# h'(T)/h(T)
gs_der_over_gs = 3*mF*(gstar*np.sqrt(geff)/heff - 1)/temps

# Backreaction on h(T) and g(T) from the Goldstone boson
# Note: we do not include backreaction in derivative
if GG == "GG":
    # Original scan (with g(T) correction calculated)
    scan_0 = '_'.join(scan.split("_")[:-1]) # get rid of GG at the end of scan name
    dir_0 = "./results/" + scan_0 + "_" + str(lam_phi) + "/g_goldstone/"
    file_0 = "gg_interp_" + str(m1) + "_" + str(kappa) + "_" + str(lam_phi) + "_" + str(ACC)
    path_0 = dir_0 + file_0
    backreaction = pd.read_csv(path_0, sep="\t", header=None)[1]
    # Add backreaction
    heff += backreaction
    geff += backreaction

# Interpolation

# Input data in ascending order 
g_fun = interp1d(temps[::-1], geff[::-1], kind='cubic')
gs_fun = interp1d(temps[::-1], heff[::-1], kind='cubic')
gs_der_over_gs_fun = interp1d(temps[::-1], gs_der_over_gs[::-1], kind='cubic')

#===========
# Functions
#===========
