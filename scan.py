'''
Script performs scan over m1 and kappa parameters in Weinberg's model

- m1: first scalar's mass in GeV
- kappa: coupling constant between phi and Higgs fields

Arguments:
- scan: scan name
- ACC: accuracy (important in Mathematica scripts)
- lam_phi: self-coupling constant for phi field
- GG: (optional) whether to include backraction in g(x) function
'''

import os
import sys
import shutil
import timeit
import numpy as np
from utils import join_line, file_append

start = timeit.default_timer()

# Arguments

scan = sys.argv[1]
ACC = int(sys.argv[2])
lam_phi = float(sys.argv[3])
GG = sys.argv[4] if len(sys.argv) == 5 else 'none'

# Paths

file_info = "./results/" + scan + "_" + str(lam_phi) + "_info"
path_results = "./results/" + scan + "_" + str(lam_phi) + "/"

if os.path.isfile(file_info):
    os.remove(file_info)

if os.path.isdir(path_results):
    shutil.rmtree(path_results)
os.makedirs(path_results)

# Scan

m1_vec = [0.5]
kappa_vec = [0.0001, 0.0002]
#m1_vec = np.round(np.linspace(0.25, 0.5, 10), 8)
#kappa_vec =np.array([0.001, 0.002])

print(m1_vec)
print(kappa_vec)

for m1 in m1_vec:
    calc_interp = 1 # whether or not to perform interpolation for MB and pBE approximations 
    for kappa in kappa_vec: 
        # Perform calculation
        print("\nScan for m1 = %s, kappa = %s\n" % (m1, kappa))
        command = join_line(["python RK4-solver.py", scan, ACC, lam_phi, kappa, m1, calc_interp, GG])
        os.system(command)
        # Update info file
        file_append(file_info, [calc_interp, kappa, m1, lam_phi, ACC])
        calc_interp = 0
    
# Running time 

time = timeit.default_timer() - start
print("\nTotal time: %s" % time)