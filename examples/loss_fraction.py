#!/usr/bin/env python3

import time

from neat.fields import stellna_qs
from neat.tracing import charged_particle_ensemble, particle_ensemble_orbit

"""                                                                           
Calculate the loss fraction of a distribution of particles
in a quasisymmetric stellarator                   
"""

nthreads_array = [1, 2, 4, 8]
r_surface_max = 0.15
r_initial = 0.1
energy = 1e4

g_field = stellna_qs.from_paper(1)
g_particle = charged_particle_ensemble(r0=r_initial, energy=energy)
print("Starting particle tracer")
threads_vs_time = []
for nthreads in nthreads_array:
    start_time = time.time()
    g_orbits = particle_ensemble_orbit(g_particle, g_field, nthreads=nthreads)
    total_time = time.time() - start_time
    print(
        f"  Running with {nthreads} threads and {g_orbits.nparticles} particles took {total_time}s"
    )
    threads_vs_time.append([nthreads, total_time])
g_orbits.loss_fraction(r_surface_max=r_surface_max)
print(f"Final loss fraction = {g_orbits.loss_fraction_array[-1]*100}%")
g_orbits.plot_loss_fraction()
