# IDP_phase_separation
Field Theoretic Simulation (FTS) code to study Liquid-Liquid Phase Separation (LLPS) of systems 
containing multiple model IDPs of same length and no net charge per molecule.

The codes in this repository could be used to generate system snapshots of Complex Langevin (CL) 
evolution at an specified interval. Then from the snapshots, thermodynamic averages of field 
operators could be computed. These codes have been used in the upcoming book chapter

   Yi-Hsuan Lin, Jonas Wessén, Tanmoy Pal, Suman Das and Hue Sun Chan, Phase-Separated Biomolecular 
   Condensates: Methods and Protocols, Pages 51-94, Springer US.
   
   https://link.springer.com/protocol/10.1007/978-1-0716-2663-4_3

The script Cl_seq_list.py contsins charge sequences of some popular model and real IDPs.
The script FTS_polyampholytes_multi_species.py contains all the routines for CL evolution.
The script submit_to_cluster.py includes a way to distribute multiple jobs to available 
CPUs in a node and uses Python's multi processing module.
The script calc_correlations.py could be used to calculate pair distribution functions.
For phase separated droplet visualization, the example script plot_snapshots.py could 
be used. Two input snapshot files can be found in the snapshots.zip file.
