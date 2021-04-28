import numpy as np
import sys
from pathlib import Path

Nx = 32
a = np.sqrt(1./6.)
dx = a
V = (Nx*dx)**3.
dv = dx*dx*dx
u0 = 0.068

def ft(x):
    return np.fft.fftn(x) * dv
def ift(x):
    return np.fft.ifftn(x) / dv

def cal_correlations(w, rho):

    w = ft(w)
    rho0_p = ft(rho[0])
    rho0_c = np.conjugate(ft(np.conjugate(rho[0])))
    rho1_p = ft(rho[1])
    rho1_c = np.conjugate(ft(np.conjugate(rho[1])))
    
    cross = rho0_p * rho1_c
    auto0 = (1J/u0) * w * rho0_c - cross
    auto1 = (1J/u0) * w * rho1_c - cross

    cross = ift(cross)/V
    auto0 = ift(auto0)/V
    auto1 = ift(auto1)/V
    
    return [cross, auto0, auto1]


cross_avg = np.zeros((Nx,Nx,Nx), dtype=complex)
auto0_avg = np.zeros((Nx,Nx,Nx), dtype=complex)
auto1_avg = np.zeros((Nx,Nx,Nx), dtype=complex)
samples = 0.

for cl_step in range(20000, 54001, 500):

    fname = '../densities/density_sv28_sv9_'+sys.argv[1]+'_step-'+str(cl_step)+'.npy'
    if Path(fname).is_file():
        print(fname)
        with open(fname, 'rb') as fs:
            w = np.load(fs)
            psi = np.load(fs)
            rho = np.load(fs)
        
        samples += 1.
        [cross, auto0, auto1] = cal_correlations(w, rho)

        cross_avg += cross
        auto0_avg += auto0
        auto1_avg += auto1

cross_avg /= samples
auto0_avg /= samples
auto1_avg /= samples

# saving to .txt files after corroborating periodic boundary conditions
f = open('correlations_run-'+sys.argv[1]+'.txt', 'w')
for i in range(Nx):
    for j in range(Nx):
        for k in range(Nx):
            r = np.sqrt(np.minimum(i, Nx-i)**2. + np.minimum(j, Nx-j)**2. + np.minimum(k, Nx-k)**2.)*dx
            np.savetxt(f, np.array([ r, cross_avg[i,j,k], auto0_avg[i,j,k], auto1_avg[i,j,k] ]).real, delimiter="  ", fmt='%10.5f', newline=' ')
            f.write('\n')

f.close()
