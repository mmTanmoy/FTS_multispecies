########################################################################################
# This piece of code imports all the routines from FTS_polyampholytes_multi_species.py #
# and runs several processes corresponding to different parameter values on the        #
# specified number of processors. Here snapshots of the system being simulated are     #
# are saved on a specified interval.                                                   #
########################################################################################

from FTS_polyampholytes_multi_species import *

import multiprocessing as mp
import time

def exe( PS , run_label ):
    np.random.seed()      # to ensure unique random seed for each process
    # CL time step
    dt = 0.0005           # CL time step in simulation
    t_intvl = 500          # CL steps interval to take snapshots
    t_tot = 60001         # total CL time steps

    # initialize fields
    init_size = 0.1
    w   = init_size * ( np.random.randn( PS.Nx,PS.Nx,PS.Nx ) + 1j * np.random.randn( PS.Nx,PS.Nx,PS.Nx ) )
    psi = init_size * ( np.random.randn( PS.Nx,PS.Nx,PS.Nx ) + 1j * np.random.randn( PS.Nx,PS.Nx,PS.Nx ) )

    w   -= np.mean(w) + 1j * np.sum(PS.rhop0s) * PS.v
    psi -= np.mean(psi)
    PS.set_fields(w,psi)

    Minv = get_M_inv( PS, dt)

    ev_f = open( 'evolution/evolution_' + run_label + '.txt' , 'w', 1)

    for t in range(1, t_tot):
        if t %t_intvl == 0:
            allQs = np.concatenate([np.array([t, t*dt]), PS.Q])
            np.savetxt(ev_f, np.concatenate([allQs.real, allQs[2:].imag]), newline='  ', delimiter='  ')
            ev_f.write('\n')
            
            with open('densities/density_'+ run_label +'_step-' + str(t) + '.npy', 'wb') as f_den:
                np.save(f_den, PS.w)
                np.save(f_den, PS.psi)
                np.save(f_den, PS.rhop)

        CL_step_SI(PS, Minv, dt, useSI=True)


if __name__ == '__main__':
    ncpus = 40
    
    lB = 5.0
    v = 0.068
    Nx = 32

    seq_list = np.array(['sv28', 'sv9'])         # sequence pairs to be simulated
    sigs, rhop0s = [], []
    for seq in seq_list:
        sig, N, the_seq = sl.get_the_charge(seq)
        sigs.append(sig)
        rhop0s.append(0.2)
    sigs, rhop0s = np.array(sigs), np.array(rhop0s) # setting up charge sequences and their bulk densities

    run_label_base = "sv28_sv9_"
    
    procs = []
    for run in range(ncpus):
        run_label =run_label_base+str(run)

        proc = mp.Process(target=exe, args=( PolySol(sigs, rhop0s, lB, v, Nx) , run_label , ) ) 
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


