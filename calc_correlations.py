#############################################################################
# The following script could be used to compute Pair Distribution Functions #
# (PDFs) from the generated field and density snapshots. It uses pandas     #
# along with numpy. For an 'n' number of indepent runs with same lB and     #
# polymer bulk densities, it out puts the file
#            'svX-svY_correlations_cross_self0_self1.txt'                   #
# in which the columns are (r, mean-of-Cross.Re, std-of-Cross.Re,           #
# mean-of-SelfX.Re, std-of-SelfX.Re, and mean-of-SelfY.Re, std-of-SelfY.Re).#
#############################################################################

import numpy as np
import sys
from pathlib import Path
import pandas as pd

# The following parameter values must be same as used in
# FTS_polyampholytes_multispecies.py/ submit_to_cluster.py
Nx = 32
a = np.sqrt(1./6.)
dx = a
V = (Nx*dx)**3.
dv = dx*dx*dx
v = 0.068

def ft(x):
    return np.fft.fftn(x) * dv
def ift(x):
    return np.fft.ifftn(x) / dv

def cal_correlations(w, rho):

    ft_w = ft(w)
    rho0_p = ft(rho[0])
    rho0_c = np.conjugate(ft(np.conjugate(rho[0]))) # FT with -Ve wave vectors
    rho1_c = np.conjugate(ft(np.conjugate(rho[1]))) # FT with -Ve wave vectors
    
    cross = rho0_p * rho1_c
    self0 = (1J/v) * ft_w * rho0_c - cross
    self1 = (1J/v) * ft_w * rho1_c - cross

    cross = ift(cross)/V
    self0 = ift(self0)/V
    self1 = ift(self1)/V
    
    return [cross, self0, self1]

# Listing all the relative distances for the periodic box
rsq = np.array([[[np.minimum(i, Nx-i)**2 + np.minimum(j, Nx-j)**2 + \
                  np.minimum(k, Nx-k)**2 for i in range(Nx)] for j \
                 in range(Nx)] for k in range(Nx)])
rs = np.sqrt(rsq)*dx
rs = rs.flatten()

seqs          = 'sv28_sv9_'  # The sequence-pair string
runs          = 40           # Number of independent runs
starting_step = 3000         # Initial CL-step of sampling
ending_step   = 5601         # Ending CL-step of sampling
interval      = 200          # Sampling interval

df_all = pd.DataFrame()

for run in range(runs):

    cross_avg = np.zeros((Nx,Nx,Nx), dtype=complex)
    self0_avg = np.zeros((Nx,Nx,Nx), dtype=complex)
    self1_avg = np.zeros((Nx,Nx,Nx), dtype=complex)
    samples = 0.

    for cl_step in range(starting_step, ending_step, interval):

        fname = 'densities/density_' + seqs + str(run) + \
            '_step-' + str(cl_step) + '.npy'
        if Path(fname).is_file():
            print(fname)
            with open(fname, 'rb') as fs:
                w = np.load(fs)
                psi = np.load(fs)
                rho = np.load(fs)
        
            samples += 1.
            [cross, self0, self1] = cal_correlations(w, rho)

            cross_avg += cross
            self0_avg += self0
            self1_avg += self1

    cross_avg /= samples
    self0_avg /= samples
    self1_avg /= samples
    
    frame = { 'r' : rs, 'cross' : cross_avg.real.flatten(),
              'self0' : self0_avg.real.flatten(),
              'self1' : self1_avg.real.flatten()}
    df = pd.DataFrame(frame)
    grouped_by_r = df.groupby('r', as_index=False).mean()
    df_all = df_all.append(grouped_by_r, ignore_index=True)

mean_std = df_all.groupby('r').agg({ 'cross':['mean', 'std'],
                                     'self0':['mean', 'std'],
                                     'self1':['mean', 'std'] })

outfile = seqs + 'correlations_cross_self0_self1.txt'
mean_std.to_csv(outfile, sep='\t', index=True, na_rep='nan', header=False)
