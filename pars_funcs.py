import math as m
from scipy.interpolate import interp1d

#============
# Parameters
#============

Mpl = 1.2209e19 # Planck mass in GeV
mF = 0.1056  # muon mass in GeV
m2 = 125 # second scalar mass (Higgs boson) in GeV

zeta_3 = 1.20205690315959429 # Riemann zeta function for arg = 3
Cz = m.sqrt(45)/128/pow(m.pi,4.5)*Mpl/mF
Cz_MB = m.sqrt(45)/512/pow(m.pi,4.5)*Mpl/mF
CY = 1/m.sqrt(45*m.pi)/512*Mpl/mF
CY_pBE = 1/m.sqrt(45*m.pi)/512*Mpl/mF

#===========
# Functions
#===========

def barJ_BE(z,barX,n):
    ''' 
    Helper functon for fun_z functions 
    Note: limit in the integral is arbitrary !!!!!!
    '''
    integral = lambda y: barX**(n+1)*y**n*m.exp(y)/(m.exp(y+z)-1.)**2
    return integrate.quad(integral,0,100)[0] # limit !!!!!!

def fun_z_MB(z,barX,barS_MB):
    ''' z' = fun_z_MB(z,barX,barS_MB) '''
    return -gs_der_over_gs_fun(barX*mF)+\
    Cz_MB*sinh(z)/m.pow(barX,5)*barS_MB*\
    (1.+barX/3.*gs_der_over_gs_fun(barX*mF))/m.sqrt(g_fun(barX*mF))

def fun_z_MB_odeint(z,barX):
    ''' z' = fun_z_MB_odeint(z,barX) '''
    return -gs_der_over_gs_fun(barX*mF)+\
    Cz_MB*sinh(z)/m.pow(barX,5)*coeff*barS_MB_interp(barX)*\
    (1.+barX/3.*gs_der_over_gs_fun(barX*mF))/m.sqrt(g_fun(barX*mF))

def fun_z_fBE(z,barX,barS_MB):
    ''' z' = fun_z_fBE(z,barX,barS_MB) '''
    return -gs_der_over_gs_fun(barX*mF)+\
    Cz_MB*sinh(z)/m.pow(barX,5)*barS_MB*zeta_3*\
    (1.+barX/3.*gs_der_over_gs_fun(barX*mF))/m.sqrt(g_fun(barX*mF))

def fun_z_pBE(z,barX,barS_pBE):
    ''' z' = fun_z_pBE(z,barX,barS_pBE) '''
    return -gs_der_over_gs_fun(barX*mF)+\
    Cz_MB*sinh(z)/m.pow(barX,5)*barS_pBE*zeta_3*\
    (1.+barX/3.*gs_der_over_gs_fun(barX*mF))/m.sqrt(g_fun(barX*mF))

def fun_z_BE(z,barX,barS_BE):
    ''' 
    z' = fun_z_BE(z,barX,barS_BE) 
    BE is related to incoming particles
    Function works for BEFD and BEBE statistics
    '''
    return (-gs_der_over_gs_fun(barX*mF)/3./barX*barJ_BE(z,barX,3)+\
    Cz*barX/m.sqrt(g_fun(barX*mF))*sinh(z)*barS_BE*\
    (1.+barX/3.*gs_der_over_gs_fun(barX*mF)))/barJ_BE(z,barX,2)

def Y_eq_MB(barX):
    ''' Y equilibrium for MB statistics '''
    return 22.5/m.pow(m.pi,4)/gs_fun(barX*mF)
    
def Y_eq_BE(barX):
    ''' Y equilibrium for BE statistics '''
    return 22.5/m.pow(m.pi,4)/gs_fun(barX*mF)*zeta_3

def Y_BE(z,barX): 
    '''
    Y(z) for BE statistics 
    Function works for BEFD and BEBE statistics
    Note: limit in the integral is arbitrary !!!!!!
    '''
    integral = lambda y: y**2/(m.exp(y+z)-1.)
    return Y_eq_MB(barX)/2.*integrate.quad(integral,0,100)[0]