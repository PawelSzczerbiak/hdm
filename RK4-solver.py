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

zeta_3 = 1.20205690315959429 # Riemann zeta function for arg = 3
Cz = m.sqrt(45) / 128/ pow(m.pi, 4.5) * Mpl / mF
Cz_MB = m.sqrt(45) / 512 / pow(m.pi, 4.5) * Mpl / mF
CY = 1 / m.sqrt(45 * m.pi) / 512 * Mpl / mF
CY_pBE = 1 / m.sqrt(45 * m.pi) / 512 * Mpl / mF

Gam1 = lam_phi * m1 / 16 / m.pi * 1000 # first scalar decay rate in GeV
coeff = 2 * kappa**2 * mF**8 / m1**4 / m2**4 # coefficient in |M|^2
barX_0 = 4 # barX = T_max / mF, starting moment for calculation
h = -0.05 # barX step
barX_end = 0.1#0.1 # ending moment for calculation

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

print("\nNodes calculation for MB and pBE statistics...")
barXs_nodes = []
nodes_MB = []
nodes_pBE = []

file_nodes = path_results + "interp_" + str(m1) + "_" + str(lam_phi) + "_" + str(int(ACC))

# Interpolate if flag is 1 or file does not exist
if calc_interp == 1 or not os.path.isfile(file_nodes):
    # Remove file if exists
    if os.path.isfile(file_nodes):
        os.remove(file_nodes)

    barXs_nodes = np.linspace(barX_end/10, barX_0, 50) # ascending order (first value possibly small)

    for  i, barX in enumerate(barXs_nodes):
        # Mathematica script execution for MB approximation
        command = join_line(['cd', path_scripts, ';', './MB_M_Weinberg', 1/barX, ACC, m1, Gam1])
        os.system(command)
        with open(path_scripts+'res_MB_M_Weinberg.dat', 'r') as f:
            nodes_MB.append((float(f.readline())))
        # Mathematica script execution for pBE approximation
        command = join_line(['cd', path_scripts, ';', './pBE_M_Weinberg', 1/barX, ACC, m1, Gam1])
        os.system(command)
        with open(path_scripts+'res_pBE_M_Weinberg.dat', 'r') as f:
            nodes_pBE.append((float(f.readline())))

        params = [barX, nodes_MB[-1], nodes_pBE[-1]]
        # Saving results
        file_append(file_nodes, params)
        print("%3d %10.2f %22.5f %22.5f" % (i, params[0], params[1], params[2]))
else:
    # --- or use read_columns_as_rows() for all colmuns
    for line in open(file_nodes, 'r'):
        elements = line.split(' ')
        barXs_nodes.append(float(elements[0]))
        nodes_MB.append(float(elements[1]))
        nodes_pBE.append(float(elements[2]))

end_nodes = timeit.default_timer()

barS_MB_interp = interp1d(barXs_nodes, nodes_MB, kind='cubic')
barS_pBE_interp = interp1d(barXs_nodes, nodes_pBE, kind='cubic')

print("-- Time: %5.2f s" % (end_nodes - start))

#======================================
# Odeint solution for MB approximation
#======================================

print("\nOdeint solution for MB...")

def Y_eq_MB(barX):
    ''' Y equilibrium for MB statistics '''
    return 22.5 / m.pow(m.pi, 4) / gs_fun(barX * mF)

def Y_eq_BE(barX):
    ''' Y equilibrium for BE statistics '''
    return Y_eq_MB(barX) * zeta_3

def fun_z_MB_odeint(z, barX):
    ''' z' = fun_z_MB_odeint(z,barX) '''
    return -gs_der_over_gs_fun(barX * mF) \
    + Cz_MB * m.sinh(z) / m.pow(barX, 5) * coeff * barS_MB_interp(barX) \
    * (1 + barX / 3 * gs_der_over_gs_fun(barX * mF)) / m.sqrt(g_fun(barX * mF))

z_0 = 0
barX_odeint = np.linspace(barX_0, barX_end, 1000) # descending order
z_odeint = integrate.odeint(fun_z_MB_odeint, z_0, barX_odeint).flatten()
Y_odeint = []
Y_eq = [] # equilibrium

# Rough estimation of the starting point for RK4 solution
found = False
for i, z in enumerate(z_odeint):
    Y_odeint.append(m.exp(-z) * Y_eq_BE(barX_odeint[i]))
    Y_eq.append(Y_eq_BE(barX_odeint[i]))
    if found == False and m.exp(-z) <= 0.95: # 5% deviation
        found = True
        z_0 = z
        barX_0 = barX_odeint[i]

print("BarX_0 = %.3f, z_0 = %.3f" % (barX_0, z_0))

#===================================
# RK4 solution for BEFD statistics
#===================================

print("\nRK4 solution for BEFD, MB, fBE, pBE...")

barXs = np.arange(barX_0, barX_end, h) # descending order

class RK4_Solver(object):
    '''
    Class which solves Boltzmann equation for pseudopotential z(barX)
    in S. Weinberg's model for BEFD case and MB, fBE, pBE approximations.
    Method: Runge-Kutta, 4th order.
    
    Parameters:
    - ACC: accuracy for Mathematica script (BEFD)
    - m1: first scalar's mass in GeV
    - Gam1: first scalar's decay rate in GeV
    - barXs: array of barX's (barX = T / mF)
    - z_0: pseudopotential for barXs[0]

    Requires:
    - global parameters: mF, coeff, Cz, Cz_MB, path_scripts
    - global functions: g_fun, gs_der_over_gs_fun, barS_MB_interp, barS_pBE_interp, Y_eq_MB, Y_eq_BE
    '''
    def __init__(self, ACC, m1, Gam1, barXs, z_0=0):
        self.ACC = ACC
        self.m1 = m1
        self.Gam1 = Gam1
        self.barXs = barXs
        self.z_0 = z_0

    # Main function

    def calculate(self):
        '''
        Calculates pseudopotential for BEFD, MB, fBE, fBE
        '''
        # first step
        z_BE, z_MB, z_fBE, z_pBE = np.full(4, self.z_0)
        self.z_ = [[z_BE, z_MB, z_fBE, z_pBE]]
        self.Y_eq_ = [Y_eq_BE(barXs[0])]
        self.Y_ = [self.Y_BE(z_BE, barXs[0]), \
        m.exp(-z_MB)*self.Y_eq_[0], m.exp(-z_fBE)*self.Y_eq_[0], m.exp(-z_pBE)*self.Y_eq_[0]]
        # nex steps
        for i in range(len(self.barXs) - 1):
            barX_prev = self.barXs[i] # previous step
            barX_curr = self.barXs[i+1] # current step
            h = barX_curr - barX_prev

            z_BE = self.calc_z(z_BE, barX_prev, h, self.calc_k_BEFD)
            z_MB = self.calc_z(z_MB, barX_prev, h, self.calc_k_MB)
            z_fBE = self.calc_z(z_fBE, barX_prev, h, self.calc_k_fBE)
            z_pBE = self.calc_z(z_pBE, barX_prev, h, self.calc_k_pBE)

            zs = [z_BE, z_MB, z_fBE, z_pBE]
            # print all simultaneously (important for checking results on the flow)
            print("%3d %10.2f %10.5f %10.5f %10.5f %10.5f" % (i, barX_curr, zs[0], zs[1], zs[2], zs[3]))
            self.z_.append(zs)

            self.Y_eq_.append(Y_eq_BE(barX_curr))
            Ys = self.Y_BE(z_BE, barX_curr), \
            m.exp(-z_MB)*self.Y_eq_[-1], m.exp(-z_fBE)*self.Y_eq_[-1], m.exp(-z_pBE)*self.Y_eq_[-1]
            self.Y_.append(Ys)

    # Helper functions

    def calc_z(self, z, barX, h, calc_k):
        ''' Calculates k coefficients for RK4 method '''
        k1 = calc_k(z, barX) 
        k2 = calc_k(z + h / 2 * k1, barX + h / 2)
        k3 = calc_k(z + h / 2 * k2, barX + h / 2)
        k4 = calc_k(z + h * k3, barX + h)
        return z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def calc_k_BEFD(self, z, barX):
        ''' Calculates k coefficient for BEFD '''
        try:
            # Mathematica script execution for BEFD
            command = join_line(['cd', path_scripts, ';', './BEFD_M_Weinberg', 1/barX, z, self.ACC, self.m1, self.Gam1])
            os.system(command)
            with open(path_scripts+'res_BEFD_M_Weinberg.dat', 'r') as f: 
                barS_BE = (float(f.readline()))
        except:
            barS_BE = None
        return self.fun_z_BE(z, barX, coeff * barS_BE) if barS_BE else 0

    def calc_k_MB(self, z, barX):
        ''' Calculates k coefficient for MB '''
        barS_MB = barS_MB_interp(barX)
        return self.fun_z_MB(z, barX, coeff * barS_MB)

    def calc_k_fBE(self, z, barX):
        ''' Calculates k coefficient for fBE '''
        barS_fBE = barS_MB_interp(barX)
        return self.fun_z_fBE(z, barX, coeff * barS_fBE)

    def calc_k_pBE(self, z, barX):
        ''' Calculates k coefficient for fBE '''
        barS_pBE = barS_pBE_interp(barX)
        return self.fun_z_pBE(z, barX, coeff * barS_pBE)
        
    def fun_z_BE(self, z, barX, barS_BE):
        ''' 
        z' = fun_z_BE(z, barX, barS_BE) 
        BE is related to incoming particles
        Function works for BEFD and BEBE statistics
        '''
        return (-gs_der_over_gs_fun(barX * mF) / 3 / barX * self.barJ_BE(z, barX, 3) \
        + Cz * barX / m.sqrt(g_fun(barX * mF)) * m.sinh(z) * barS_BE \
        * (1 + barX / 3 * gs_der_over_gs_fun(barX * mF))) / self.barJ_BE(z, barX, 2)

    def barJ_BE(self, z, barX, n):
        ''' 
        Helper functon for fun_z_BE() function 
        Note: limit in the integral is arbitrary !!!!!!
        '''
        integral = lambda y: barX**(n + 1) * y**n * m.exp(y) / (m.exp(y + z) - 1)**2
        return integrate.quad(integral, 0, 100)[0] # limit !!!!!!

    def fun_z_MB(self, z, barX, barS_MB):
        ''' z' = fun_z_MB(z, barX, barS_MB) '''
        return -gs_der_over_gs_fun(barX * mF) \
        + Cz_MB * m.sinh(z) / m.pow(barX, 5) * barS_MB \
        * (1 + barX / 3 * gs_der_over_gs_fun(barX * mF)) / m.sqrt(g_fun(barX * mF))

    def fun_z_fBE(self, z, barX, barS_MB):
        ''' z' = fun_z_fBE(z, barX, barS_MB) '''
        return -gs_der_over_gs_fun(barX * mF) \
        + Cz_MB * m.sinh(z) / m.pow(barX, 5) * barS_MB * zeta_3 \
        * (1 + barX / 3 * gs_der_over_gs_fun(barX * mF)) / m.sqrt(g_fun(barX * mF))

    def fun_z_pBE(self, z, barX, barS_pBE):
        ''' z' = fun_z_pBE(z, barX, barS_pBE) '''
        return -gs_der_over_gs_fun(barX * mF) \
        + Cz_MB * m.sinh(z) / m.pow(barX, 5) * barS_pBE * zeta_3 \
        * (1 + barX / 3 * gs_der_over_gs_fun(barX * mF)) / m.sqrt(g_fun(barX * mF))

    def Y_BE(self, z, barX): 
        '''
        Y(z) for BE statistics 
        Function works for BEFD and BEBE statistics
        Note: limit in the integral is arbitrary !!!!!!
        '''
        integral = lambda y: y**2 / (m.exp(y + z) - 1)
        return Y_eq_MB(barX) / 2 * integrate.quad(integral, 0, 100)[0]

S = RK4_Solver(ACC, m1, Gam1, barXs)
S.calculate()

end_RK4 = timeit.default_timer()
print("-- Time: %5.2f s" % (end_RK4 - start))

#=======
# Plots
#=======

plt.plot(barX_odeint, Y_odeint, color='blue', label=r'$Y_{\rm odeint}$')
plt.plot(barX_odeint, Y_eq, color='grey', label=r'$Y_{\rm eq}$')
plt.xlabel(r'$\bar{x}=T/m_F$')
plt.ylabel(r'$Y$')
plt.legend(loc='upper right')
plt.savefig(path_results + "y_odeint.pdf")
plt.close()

# TODO: 
# - plot Y
# - refactor Y
# - save z, Y
# - calc g
# - GG = True