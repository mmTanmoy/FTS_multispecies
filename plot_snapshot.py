# The following is an example script to demonstrate how FTS density snapshots #
# could be visualized using the python package 'mayavi' from mlab.            #
#          https://docs.enthought.com/mayavi/mayavi/                          #

from mayavi import mlab
import numpy as np
import sys

Nx = 32  # Must be same as in FTS_polyampholytes_multi_species.py
x, y, z = np.mgrid[0:Nx, 0:Nx, 0:Nx]

# for center-of-mass calculations:
psi = 2.*np.pi / Nx * np.array([ x , y , z])
xi   = np.cos( psi )
zeta = np.sin( psi )

# Get center of mass of clipped real(rho), where rho has (Nx,Nx,Nx) shape,
# assuming periodic boundary conditions.
# See https://en.wikipedia.org/wiki/Center_of_mass for details
def get_com(rho):
    rho_R = rho.real.clip(min=0)
    M_tot = np.sum(rho_R)
    
    av_xi   = np.array( [ np.sum( rho_R * xi[i] ) for i in range(0,3) ] ) / M_tot
    av_zeta = np.array( [ np.sum( rho_R * zeta[i] ) for i in range(0,3) ] ) / M_tot
    av_phi  = np.array( [ np.arctan2( av_zeta[i] , av_xi[i] ) for i in range(0,3) ] )
    av_phi  = (av_phi + 2. * np.pi ) % (2. * np.pi)
    com     = np.round( Nx * av_phi / ( 2. * np.pi ) ).astype(int)
    print(com)
    return com

# Shift density profiles to their center of mass coordinate system
def shift( rho, com ):
    rho_new = np.zeros(rho.shape,dtype=complex)
    for i in range(Nx):
        ii = int( (i+Nx/2+com[0])%Nx )
        for j in range(Nx):
            jj = int( (j+Nx/2+com[1])%Nx )
            for k in range(Nx):
                kk = int( (k+Nx/2+com[2])%Nx )
                rho_new[i,j,k] = rho[ii,jj,kk]
                
    return rho_new

name = 'density_sv28_'+sys.argv[1]+'_39_step-52000.npy' # Input snapshot file
with open(name, 'rb') as f:
    w = np.load(f)
    psi = np.load(f)
    rhos = np.load(f)

com0 = get_com(rhos[0])
rho0 = shift(rhos[0], com0)
com1 = get_com(rhos[1])
rho1 = shift(rhos[1], com1)

# To visualize co-phase separated sequences, chose a large vmax to make the
# plots transparent and plot multiple times. Otherwise, the last plotted
# sequence may block the previous ones.

vmax = 30.
mlab.figure( bgcolor=(0, 0, 0), size = (500,500) )
mlab.view(65, 90)
for i in range(7):
        mlab.pipeline.volume( mlab.pipeline.scalar_field(rho0.real.clip(min=0)) , vmin=0.0, vmax=vmax, color=(1,0,0) )
        mlab.pipeline.volume( mlab.pipeline.scalar_field(rho1.real.clip(min=0)) , vmin=0.0, vmax=vmax, color=(0,0,1) )

mlab.title('sv28-'+sys.argv[1],color=(1,1,1),height=0.9, size=0.7)
mlab.outline( color=(1,1,1) )
mlab.show()
#mlab.savefig('sv28-sv24.png')
