###########################################################################
# Field theoretic simulation (FTS) code for a multi-species polyampholyte #
# solution. The code contains routines for Complex-Langevin evolution     #
# using a semi-implicit integration scheme, and for computations of the   #
# chemical potential and osmotic pressure. The code is used in the        #
# publication                                                             #
#                                                                         #
#    Lin Y, Wessén J, Pal T, Das S & Chan H S, XXXXX (2021)               #
#                                                                         #
# and follows the methods described therein. The main function shows      # 
# code's intended usage.                                                  #
###########################################################################

import numpy as np
from numpy.fft import fftn as ft
from numpy.fft import ifftn as ift
import CL_seq_list as sl               # cobtains charge sequences of several IDPs

#----------------------- Define polymer solution as a class object -----------------------
class PolySol:
    def __init__( self, sigmas, rhop0s, lB, v, Nx):
        self.lB    = lB                  # Reduced Bjerrum length 
        self.sigs  = sigmas              # Charge sequences
        self.rhop0s = rhop0s             # Bulk polymer bead densities
        self.n_species = self.rhop0s.shape[0] # Total number of species
        self.N     = np.array([ self.sigs[seq].shape[0] for seq in range(self.n_species)])   # Length of polyampholytes
        self.v     = v                   # Excluded volume parameter
        self.a     = 1./np.sqrt(6.)      # Smearing length
        self.Nx    = Nx                  # Number of grid points: the resolution
        self.dx    = self.a              # distance between two n.n. grid points
        self.L     = self.dx*Nx          # Box edge length
        self.V     = self.L**3           # Box volume
        self.dV    = self.dx**3          # delta volume of each grid 

        # wave number vectors of the grid space
        ks1d    = 2.*np.pi*np.fft.fftfreq(self.Nx,self.dx) # k's in 1D reciprocal space
        self.kz = np.tile(ks1d, (self.Nx,self.Nx,1)) # 3D array with kz[i,j,l] = ksld[l]
        self.kx = np.swapaxes( self.kz, 0, 2) # 3D array with kx[i,j,l] = ksld[i]
        self.ky = np.swapaxes( self.kz, 1, 2) # 3D array with ky[i,j,l] = ksld[j]
        self.k2 = self.kx*self.kx + self.ky*self.ky + self.kz*self.kz # 3D array of k*k
         
        self.Gamma   = np.exp(-self.k2*self.a**2/2.)    # Gaussian smearing
        self.Prop    = np.exp(-self.k2/6. )     # Gaussian chain n.n propagator

        self.GT2_w   = ( self.k2*self.a**2/3. - 1./2. )*self.Gamma  # smearing in pressure
        self.GT2_psi = ( self.k2*self.a**2/3. - 1./6. )*self.Gamma  # smearing in pressure


        # Gaussian chain correlation functions in the k-space
        Gij = np.array([ np.exp(-np.tensordot(np.arange(self.N[seq]), self.k2, axes=0)/6) for seq in range(self.n_species) ])

        Mcc = np.array([ np.kron(self.sigs[seq], self.sigs[seq]).reshape((self.N[seq], self.N[seq])) for seq in range(self.n_species) ])
        Tcc = np.array([ np.array([ np.sum(Mcc[seq].diagonal(n) + Mcc[seq].diagonal(-n)) for n in range(self.N[seq])]) for seq in range(self.n_species) ])

        Tmm = np.array([ 2*np.arange(self.N[seq],0,-1) for seq in range(self.n_species) ])
 
        Mmc = np.array([ np.kron(self.sigs[seq], np.ones(self.N[seq])).reshape((self.N[seq], self.N[seq])) for seq in range(self.n_species) ])
        Tmc = np.array([ np.array([ np.sum(Mmc[seq].diagonal(n) + Mmc[seq].diagonal(-n)) for n in range(self.N[seq])]) for seq in range(self.n_species) ])

        for seq in range(self.n_species):
            Tcc[seq,0] /= 2.
            Tmm[seq,0] /= 2.
            Tmc[seq,0] /= 2.
            
        self.gcc = np.array([ Gij[seq].T.dot(Tcc[seq]).T / self.N[seq] for seq in range(self.n_species) ])
        self.gmm = np.array([ Gij[seq].T.dot(Tmm[seq]).T / self.N[seq] for seq in range(self.n_species) ])
        self.gmc = np.array([ Gij[seq].T.dot(Tmc[seq]).T / self.N[seq] for seq in range(self.n_species) ])

        # Fields
        self.w   = np.zeros( ( self.Nx, self.Nx, self.Nx ), dtype=complex )
        self.psi = np.zeros( ( self.Nx, self.Nx, self.Nx ), dtype=complex )

        # Single polymer partition function
        self.Q = np.ones(self.n_species, dtype=complex)

        # Chain propagators, this works for polymer species of same lenghts only
        self.qF = np.zeros( ( self.n_species, self.N[0], self.Nx, self.Nx, self.Nx ), dtype=complex )
        self.qB = np.zeros( ( self.n_species, self.N[0], self.Nx, self.Nx, self.Nx ), dtype=complex )

        # Field operators for the bead- and charge density
        self.rhop = np.zeros( ( self.n_species, self.Nx, self.Nx, self.Nx ), dtype=complex )
        self.rhoc = np.zeros( ( self.n_species, self.Nx, self.Nx, self.Nx ), dtype=complex )

    # taking Laplacian of x via Fourier transformation
    def lap(self, x):
        return -ift( self.k2 * ft( x ) ) 
    
    # Obtain densities from fields
    def calc_densities( self ):

        w_s   = ift( self.Gamma*ft( self.w )   )
        psi_s = ift( self.Gamma*ft( self.psi ) )    

        for seq in range(self.n_species):
        
            PSI =  1j*( np.tensordot( np.ones(self.N[seq]), w_s   , axes=0 ) + np.tensordot( self.sigs[seq], psi_s , axes=0) )

            self.qF[seq, 0]  = np.exp( -PSI[0]  )
            self.qB[seq, -1] = np.exp( -PSI[-1] )
    
            for i in range( self.N[seq]-1 ):
                # forwards propagator
                self.qF[seq, i+1] = np.exp( -PSI[i+1] )*ift( self.Prop*ft(self.qF[seq, i]) )
                # backwards propagator
                j = self.N[seq]-i-1
                self.qB[seq, j-1] = np.exp( -PSI[j-1] )*ift( self.Prop*ft(self.qB[seq, j]) )

            self.Q[seq] = np.sum( self.qF[seq, -1] )  * self.dV / self.V
            qs = self.qF[seq] * self.qB[seq] * np.exp(PSI) 

            self.rhop[seq] = self.rhop0s[seq] / self.N[seq] / self.Q[seq] * np.sum(qs, axis=0)
            self.rhoc[seq] = self.rhop0s[seq] / self.N[seq] / self.Q[seq] * qs.T.dot(self.sigs[seq]).T 

    # returns the polymer chemical potential for the current field configuration
    def get_chem_pot( self ):
        mu_p = np.array([ np.log( self.rhop0s[seq] / self.N[seq] ) - np.log( self.Q[seq] ) for seq in range(self.n_species) ])
        return mu_p

    def get_pressure( self ):
        ft_w   = ft( self.w )
        ft_psi = ft( self.psi )

        w_s   = ift( self.Gamma * ft_w   )
        psi_s = ift( self.Gamma * ft_psi )

        Pi = 0. + 1j*0.
        for seq in range(self.n_species):
            PSI = 1j*( np.tensordot( np.ones(self.N[seq]) , w_s   , axes=0 ) + np.tensordot( self.sigs[seq], psi_s , axes=0 ) )

            qs = self.qB[seq] * self.qF[seq] * np.exp( PSI ) / self.Q[seq]

            lap_qB = np.array( [ self.lap( np.exp( PSI[i] ) * self.qB[seq,i] ) for i in range(self.N[seq])] )
            term1  = np.sum( self.qF[seq] * lap_qB ) / ( 9.*self.Q[seq] )

            term2  = 1j*np.sum( np.array([ qs[i] * ift( self.GT2_w * ft_w ) for i in range(self.N[seq]) ]) )
            term3  = 1j*np.sum( np.array([ qs[i] * self.sigs[seq,i] * ift( self.GT2_psi * ft_psi ) for i in range(self.N[seq]) ]) )

            Pi += (self.rhop0s[seq] / self.N[seq]) * (1.0 - ( term1 + term2 + term3 ) * self.dV / self.V)

        return Pi

    # Use this function to set/initialise the fields. 
    def set_fields(self, w, psi):
        self.w   = w
        self.psi = psi
        self.calc_densities()
 
#---------------------------- Complex Langevin Time Evolution ----------------------------

# Semi-implicit method 
def CL_step_SI(PS, M_inv, dt, useSI=True):

    std     = np.sqrt( 2 * dt / PS.dV )
    eta_w   = std*np.random.randn( PS.Nx, PS.Nx, PS.Nx )
    eta_psi = std*np.random.randn( PS.Nx, PS.Nx, PS.Nx )
 
    dw   = -dt*( np.sum( np.array([ 1j*ift( PS.Gamma*ft( PS.rhop[seq] ) ) for seq in range(PS.n_species) ]), axis=0) + PS.w/PS.v                         ) + eta_w
    dpsi = -dt*( np.sum( np.array([ 1j*ift( PS.Gamma*ft( PS.rhoc[seq] ) ) for seq in range(PS.n_species) ]), axis=0) - PS.lap(PS.psi) / (4.*np.pi*PS.lB) ) + eta_psi
    
    if useSI: # Semi-implicit CL step
        ft_dw, ft_dpsi = ft( dw ) , ft( dpsi )
        dw_tmp   = M_inv[0,0] * ft_dw + M_inv[0,1] * ft_dpsi
        dpsi_tmp = M_inv[1,0] * ft_dw + M_inv[1,1] * ft_dpsi
 
        PS.w   += ift( dw_tmp ) 
        PS.psi += ift( dpsi_tmp ) 
    else:  # Euler CL step
        PS.w   += dw
        PS.psi += dpsi

    PS.calc_densities()
 
# get M_inv for semi-implicit CL integration method
def get_M_inv( PS, dt ):
    K11 = PS.Gamma**2 * np.sum( np.array([ PS.rhop0s[seq] * PS.gmm[seq] for seq in range(PS.n_species) ]), axis=0) + 1. / PS.v
    K12 = PS.Gamma**2 * np.sum( np.array([ PS.rhop0s[seq] * PS.gmc[seq] for seq in range(PS.n_species) ]), axis=0)
    K22 = PS.Gamma**2 * np.sum( np.array([ PS.rhop0s[seq] * PS.gcc[seq] for seq in range(PS.n_species) ]), axis=0) + PS.k2 / (4.*np.pi*PS.lB)
    K11[0,0,0] = 1. / PS.v
  
    M = np.array( [ [ 1.+dt*K11 , dt*K12 ] , [ dt*K12 , 1.+dt*K22 ] ]  )
    det_M = M[0,0] * M[1,1] - M[0,1] * M[1,0]
    M_inv = np.array( [ [ M[1,1] , - M[0,1] ] , [ - M[1,0] , M[0,0] ] ] ) / det_M

    return M_inv
