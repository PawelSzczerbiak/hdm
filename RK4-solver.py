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
import scipy.special as spec
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

# endings for result files
file_nodes_info =join_line([m1, lam_phi, ACC], "_") # nodes
file_info =join_line([m1, kappa, lam_phi, ACC], "_") # Y, z

# Parameters 

Mpl = 1.2209e19 # Planck mass in GeV
mF = 0.1056  # muon mass in GeV
mP = 0.13957 # pion mass in GeV
m2 = 125 # second scalar mass (Higgs boson) in GeV

zeta_3 = 1.20205690315959429 # Riemann zeta function for arg = 3
Cz = np.sqrt(45) / 128/ pow(np.pi, 4.5) * Mpl / mF
Cz_MB = np.sqrt(45) / 512 / pow(np.pi, 4.5) * Mpl / mF

Gam1 = lam_phi * m1 / 16 / np.pi # first scalar decay rate in GeV
Gam2 = 0.004  # second scalar decay rate in GeV
coeff = 2 * kappa**2 * mF**8 / m1**4 / m2**4 # coefficient in |M|^2
barX_0 = 4 # barX = T_max / mF, starting moment for calculation
h = -0.015 # barX step
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
    backreaction = read_columns_as_rows(path_0, [1])
    # Add backreaction
    if(backreaction):
        heff += backreaction[0]
        geff += backreaction[0]
    else:
        print("\n--- WARNING: File with GG corrections not found!")

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

# barS in MB approximation for ACC = -3
def barS_MB_ACC3(barX, m1, Gam1):
    x = 1/barX
    alfa = m1**2*m2**2/(m2**2 - m1**2)**2/mF**4*(m1*Gam2 - m2*Gam1)**2
    bs = m1**2/mF**2
    if m1 <= 2*mP:
        barS_MB = m1**3*m2**4/mF**6/Gam1*np.pi*(bs - 4)*(bs**2 + alfa)/ \
        ((bs - m2**2/mF**2)**2 + Gam2**2*m2**2/mF**4)*np.sqrt(bs-4)*spec.kv(1, x*np.sqrt(bs))
    else:
        # additional annihilation into pion pair
        barS_MB = m1**3*m2**4/mF**6/Gam1*np.pi*bs**(3/2)* \
        ((1 - 4*mF**2/m1**2)**(3/2) + 1/27*(m1/mF)**2*(1 + 11/2*mP**2/m1**2)**2*np.sqrt(1 - 4*mP**2/m1**2))*\
        (bs**2)/((bs - m2**2/mF**2)**2 + Gam2**2*m2**2/mF**4)*spec.kv(1, x*np.sqrt(bs))
    return barS_MB

file_nodes = path_results + "interp_" + file_nodes_info

if ACC == -3: # the worst approximation (simple algebraic computation)
    barXs_nodes = np.linspace(barX_end/10, barX_0, 50) # ascending order (first value possibly small)
    for barX in barXs_nodes:
        nodes_MB.append(barS_MB_ACC3(barX, m1, Gam1))
    nodes_pBE = nodes_MB/np.tanh(m1/(4*mF*barXs_nodes))/zeta_3**2
else: # integration needed
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

            params = [round(barX, 5), nodes_MB[-1], nodes_pBE[-1]]
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
    return 22.5 /np.pi**4 / gs_fun(barX * mF)

def Y_eq_BE(barX):
    ''' Y equilibrium for BE statistics '''
    return Y_eq_MB(barX) * zeta_3

def fun_z_MB_odeint(z, barX):
    ''' z' = fun_z_MB_odeint(z,barX) '''
    return -gs_der_over_gs_fun(barX * mF) \
    + Cz_MB * m.sinh(z) / barX**5 * coeff * barS_MB_interp(barX) \
    * (1 + barX / 3 * gs_der_over_gs_fun(barX * mF)) / np.sqrt(g_fun(barX * mF))

z_0 = 0
barX_odeint = np.linspace(barX_0, barX_end, 1000) # descending order
z_odeint = integrate.odeint(fun_z_MB_odeint, z_0, barX_odeint).flatten()
Y_odeint = []
Y_eq = [] # equilibrium

# Rough estimation of the starting point for RK4 solution
found = False
for i, z in enumerate(z_odeint):
    Y_odeint.append(np.exp(-z) * Y_eq_BE(barX_odeint[i]))
    Y_eq.append(Y_eq_BE(barX_odeint[i]))
    if found == False and np.exp(-z) <= 0.99: # 1% deviation
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
        Calculates pseudopotential z_, normalized relic density Y_ and Y_eq_
        for BEFD, MB, fBE, fBE
        '''
        # first step
        z_BE, z_MB, z_fBE, z_pBE = np.full(4, self.z_0)
        self.z_ = [[z_BE, z_MB, z_fBE, z_pBE]]
        self.Y_ = [self.calc_Y(self.z_[0], barXs[0])]
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
            self.z_.append(zs)

            Ys = self.calc_Y(zs, barX_curr)
            self.Y_.append(Ys)

            # print all simultaneously (important for checking results on the flow)
            print("%3d %10.2f    z: %10.5f %10.5f %10.5f %10.5f    Y: %10.5f %10.5f %10.5f %10.5f" \
            % (i, barX_curr, zs[0], zs[1], zs[2], zs[3], Ys[0], Ys[1], Ys[2], Ys[3]))

    # Helper functions

    def calc_z(self, z, barX, h, calc_k):
        ''' 
        Calculates k coefficients for RK4 method 
        h is important because barXs data may be unevenly distributed
        '''
        k1 = calc_k(z, barX) 
        k2 = calc_k(z + h / 2 * k1, barX + h / 2)
        k3 = calc_k(z + h / 2 * k2, barX + h / 2)
        k4 = calc_k(z + h * k3, barX + h)
        return z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def calc_Y(self, zs, barX):
        ''' Calculates Y values for a given zs array and barX''' 
        Ys =[self.Y_BE(zs[0], barX)]
        Ys.extend(np.multiply(np.exp(np.negative(zs[1:])), Y_eq_BE(barX)))
        return Ys

    def calc_k_BEFD(self, z, barX):
        ''' Calculates k coefficient for BEFD '''
        if ACC == -3: # the worst approximation (simple algebraic computation)
            x = 1/barX
            alfa = m1**2*m2**2/(m2**2 - m1**2)**2/mF**4*(m1*Gam2 - m2*Gam1)**2
            if m1 <= 2*mP:
                barS_BE = np.exp(-2*z)*x**4*m1**7*m2**4/mF**10/Gam1*np.pi*\
                (1 - 4*mF**2/m1**2)**(3/2)*(m1**4/mF**4 + alfa)/ \
                ((m1**2/mF**2 - m2**2/mF**2)**2 + Gam2**2*m2**2/mF**4)*1/2/x*mF/m1*spec.kv(1, (m1*x)/mF)
            else:
                # additional annihilation into pion pair
                barS_BE = np.exp(-2*z)*x**4*m1**7*m2**4/mF**10/Gam1*np.pi*\
                ((1 - 4*mF**2/m1**2)**(3/2) + 1/27*(m1/mF)**2*(1 + 11/2*mP**2/m1**2)**2* \
                np.sqrt(1 - 4*mP**2/m1**2)/np.tanh((m1*x)/(4*mF))**2)*(m1**4/mF**4)/ \
                ((m1**2/mF**2 - m2**2/mF**2)**2 + Gam2**2*m2**2/mF**4)*1/2/x*mF/m1*spec.kv(1, (m1*x)/mF)
        else: # intergration needed
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
        + Cz * barX / np.sqrt(g_fun(barX * mF)) * m.sinh(z) * barS_BE \
        * (1 + barX / 3 * gs_der_over_gs_fun(barX * mF))) / self.barJ_BE(z, barX, 2)

    def barJ_BE(self, z, barX, n):
        ''' 
        Helper functon for fun_z_BE() function 
        Note: limit in the integral is arbitrary !!!!!!
        '''
        integral = lambda y: barX**(n + 1) * y**n * np.exp(y) / (np.exp(y + z) - 1)**2
        return integrate.quad(integral, 0, 100)[0] # limit !!!!!!

    def fun_z_MB(self, z, barX, barS_MB):
        ''' z' = fun_z_MB(z, barX, barS_MB) '''
        return -gs_der_over_gs_fun(barX * mF) \
        + Cz_MB * m.sinh(z) / barX**5 * barS_MB \
        * (1 + barX / 3 * gs_der_over_gs_fun(barX * mF)) / np.sqrt(g_fun(barX * mF))

    def fun_z_fBE(self, z, barX, barS_MB):
        ''' z' = fun_z_fBE(z, barX, barS_MB) '''
        return -gs_der_over_gs_fun(barX * mF) \
        + Cz_MB * m.sinh(z) / barX**5 * barS_MB * zeta_3 \
        * (1 + barX / 3 * gs_der_over_gs_fun(barX * mF)) / np.sqrt(g_fun(barX * mF))

    def fun_z_pBE(self, z, barX, barS_pBE):
        ''' z' = fun_z_pBE(z, barX, barS_pBE) '''
        return -gs_der_over_gs_fun(barX * mF) \
        + Cz_MB * m.sinh(z) / barX**5 * barS_pBE * zeta_3 \
        * (1 + barX / 3 * gs_der_over_gs_fun(barX * mF)) / np.sqrt(g_fun(barX * mF))

    def Y_BE(self, z, barX): 
        '''
        Y(z) for BE statistics 
        Function works for BEFD and BEBE statistics
        Note: limit in the integral is arbitrary !!!!!!
        '''
        integral = lambda y: y**2 / (np.exp(y + z) - 1)
        return Y_eq_MB(barX) / 2 * integrate.quad(integral, 0, 100)[0]

S = RK4_Solver(ACC, m1, Gam1, barXs, z_0)
S.calculate()
z_RK4 = S.z_
Y_RK4 = S.Y_

end_RK4 = timeit.default_timer()
print("-- Time: %5.2f s" % (end_RK4 - end_nodes))

#=======
# Plots
#=======

# z
plt.plot(barXs, [z[0] for z in z_RK4], color='red', label=r'$z_{\rm BEFD}$')
plt.plot(barXs, [z[1] for z in z_RK4], color='blue', label=r'$z_{\rm MB}$')
plt.plot(barXs, [z[2] for z in z_RK4], color='magenta', label=r'$z_{\rm fBE}$')
plt.plot(barXs, [z[3] for z in z_RK4], color='green', label=r'$z_{\rm pBE}$')
plt.xlabel(r'$\bar{x}=T/m_F$')
plt.ylabel(r'$z$')
plt.legend(loc='upper right')
plt.savefig(path_results + "z_" + file_info + ".pdf")
plt.close()

# Y
plt.plot(barX_odeint, Y_eq, color='grey', label=r'$Y_{\rm eq}$')
plt.plot(barX_odeint, Y_odeint, color='cyan', label=r'$Y_{\rm odeint}$')
plt.plot(barXs, [Y[0] for Y in Y_RK4], color='red', label=r'$Y_{\rm BEFD}$')
plt.plot(barXs, [Y[1] for Y in Y_RK4], color='blue', label=r'$Y_{\rm MB}$')
plt.plot(barXs, [Y[2] for Y in Y_RK4], color='magenta', label=r'$Y_{\rm fBE}$')
plt.plot(barXs, [Y[3] for Y in Y_RK4], color='green', label=r'$Y_{\rm pBE}$')
plt.xlabel(r'$\bar{x}=T/m_F$')
plt.ylabel(r'$Y$')
plt.legend(loc='upper right')
plt.savefig(path_results + "Y_" + file_info + ".pdf")
plt.close()

#======
# Save
#======

np.savetxt(path_results + "barX_" + file_info, barXs, fmt='%.5f')
np.savetxt(path_results + "z_" + file_info, z_RK4, fmt='%.10f')
np.savetxt(path_results + "Y_" + file_info, Y_RK4, fmt='%.10f')

# TODO: 
# - calc g(T)