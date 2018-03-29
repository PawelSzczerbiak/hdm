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
from utils import join_line, file_append, read_columns_as_rows
from pars_funcs import * # parameters and functions (safe import)

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
# (the rest is imported from pars_funcs.py)

Gam1 = lam_phi*m1/16/m.pi*1000 # first scalar decay rate in GeV
coeff = 2*kappa**2*mF**8/m1**4/m2**4 # coefficient in |M|^2
barX_max = 4 # T_max/mF, starting moment for RK4 method

# Paths

path_results = "./results/" + scan + "_" + str(lam_phi) + "/"
path_scripts = "./scripts/"
path_data = "./data/"

#==========================================================================
# Read h(T), g(T) and gstar(T) functions from data folder (see 1503.03513)
#==========================================================================

heff = read_columns_as_rows(path_data + "heffdhs.dat", [1]) # take only the second column
heff = heff[0] # retrieve pure row without packing in []
geff = read_columns_as_rows(path_data + "geffdhs.dat", [1])
geff = geff[0] # retrieve pure row without packing in []
df_gstar = read_columns_as_rows(path_data + "gstardhs.dat", [0,1])
temps = df_gstar[0] # temperatures
gstar = df_gstar[1]

# Using pandas (slightly less efficient but instructive)

#heff = pd.read_csv(path_data + "heffdhs.dat", sep="\t", header=None)[1] # take only the second column
#geff = pd.read_csv(path_data + "geffdhs.dat", sep="\t", header=None)[1]
#df_gstar = pd.read_csv(path_data + "gstardhs.dat", sep="\t", names = ['T', 'gstar'])
#temps = df_gstar['T'].values
#gstar = df_gstar['gstar'].values

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
    # backreaction = pd.read_csv(path_0, sep="\t", header=None)[1]
    backreaction = read_columns_as_rows(path_0, [1])[0]
    # Add backreaction
    heff += backreaction
    geff += backreaction

# Interpolation

# Input data must be in ascending order 
g_fun = interp1d(temps[::-1], geff[::-1], kind='cubic')
gs_fun = interp1d(temps[::-1], heff[::-1], kind='cubic')
gs_der_over_gs_fun = interp1d(temps[::-1], gs_der_over_gs[::-1], kind='cubic')

#=========================================
# Interpolation for MB and pBE statistics
#=========================================

print("Nodes calculation for MB and pBE statistics...")
barX_vec = []
nodes_MB = []
nodes_pBE = []

file_nodes = path_results+"interp_"+str(m1)+"_"+str(lam_phi)+"_"+str(int(ACC))

# Interpolate if flag is 1 or file does not exist
if calc_interp == 1 or not os.path.isfile(file_nodes):

    # Remove file if exists
    if os.path.isfile(file_nodes):
        os.remove(file_nodes)

    barX_vec = np.linspace(0.01, barX_max, 6)

    for  i, barX in enumerate(barX_vec):
        x = 1./barX
        # Mathematica script execution for MB approximation
        command = join_line(['cd', path_scripts, ';' , './MB_M_Weinberg', x, ACC, m1, Gam1])
        os.system(command)
        with open(path_scripts+'res_MB_M_Weinberg.dat', 'r') as f:
            nodes_MB.append((float(f.readline())))
        # Mathematica script execution for pBE approximation
        command = join_line(['cd', path_scripts, ';' , './pBE_M_Weinberg', x, ACC, m1, Gam1])
        os.system(command)
        with open(path_scripts+'res_pBE_M_Weinberg.dat', 'r') as f:
            nodes_pBE.append((float(f.readline())))

        params = [barX, nodes_MB[-1], nodes_pBE[-1]]
        # Saving results
        file_append(file_nodes, params)
        print(i, barX, params)
else:
    # --- or use read_columns_as_rows() for all colmuns
    for line in open(file_nodes, 'r'):
        elements = line.split(' ')
        barX_vec.append(float(elements[0]))
        nodes_MB.append(float(elements[1]))
        nodes_pBE.append(float(elements[2]))