########################################################################################
# This piece of code imports all the routines from FTS_polyampholytes_multi_species.py #
# and runs several processes corresponding to different parameter values on the        #
# specified number of processors. Here snapshots of the system being simulated are     #
# are saved on a specified interval.                                                   #
########################################################################################

from FTS_polyampholytes_multi_species import *
import multiprocessing as mp
import CL_seq_list as sl               # cobtains charge sequences of several IDPs

# Target function for multi-processing module
def exe( PS, dt, t_intvl, t_tot, run_label ):
    np.random.seed()      # to ensure unique random seed for each process

    # initialize fields
    init_size = 0.1
    w   = init_size * ( np.random.randn( PS.Nx,PS.Nx,PS.Nx ) + 1j * np.random.randn( PS.Nx,PS.Nx,PS.Nx ) )
    psi = init_size * ( np.random.randn( PS.Nx,PS.Nx,PS.Nx ) + 1j * np.random.randn( PS.Nx,PS.Nx,PS.Nx ) )

    w   -= np.mean(w) + 1j * np.sum(PS.rhop0s) * PS.v
    psi -= np.mean(psi)
    PS.set_fields(w,psi)

    Minv = get_M_inv( PS, dt)

    ev_f = open( 'evolution/evolution_' + run_label + '.txt' , 'w', 1)

    for t in range(t_tot):
        if t %t_intvl == 0:

            np.savetxt(ev_f, np.concatenate([ [t, t*dt], PS.Q.real, PS.Q.imag ]), newline='\t', delimiter='\t')
            ev_f.write('\n')
            
            with open('densities/density_'+ run_label +'_step-' + str(t) + '.npy', 'wb') as f_den:
                np.save(f_den, PS.w)
                np.save(f_den, PS.psi)
                np.save(f_den, PS.rhop)
            f_den.close()

        CL_step_SI(PS, Minv, dt, useSI=True)
    
    ev_f.close()


if __name__ == '__main__':

    ncpus = 40           # Number of CPUs to be used
    dt = 0.005           # CL time step in simulation
    t_intvl = 200        # CL steps interval to take snapshots
    t_tot = 60001        # total CL time steps
    
    lB = 5.0             # Bjerrum length
    v = 0.068            # Excluded volume parameter 
    Nx = 32              # Grid dimension Nx*Nx*Nx
    bulk_poly = 0.2      # Polymer bulk densities

    # sequence pairs to be simulated
    sig1, _, _ = sl.get_the_charge('sv28')
    sig2, _, _ = sl.get_the_charge('sv9')

    # setting up charge sequences and their bulk densities
    sigs = np.array([ sig1, sig2 ])
    rhop0s = np.array([ bulk_poly, bulk_poly ])

    run_label_base = "sv28_sv9_"
    
    procs = []
    for run in range(ncpus):
        run_label =run_label_base+str(run)

        proc = mp.Process(target=exe, args=( PolySol(sigs, rhop0s, lB, v, Nx), dt, t_intvl, t_tot, run_label, ) ) 
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
