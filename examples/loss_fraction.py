#!/usr/bin/env python3

import time

from neat.gyronimo.fields import stellna_qs
from neat.gyronimo.tracing import charged_particle_ensemble, particle_ensemble_orbit

"""                                                                           
Calculate the loss fraction of a distribution of particles
in a quasisymmetric stellarator                   
"""

nthreads = 8
r_surface_max = 0.13

g_field = stellna_qs.from_paper(4)
g_particle = charged_particle_ensemble()
start_time = time.time()
print("Starting particle tracer")
g_orbits = particle_ensemble_orbit(g_particle, g_field, nthreads=nthreads)
total_time = time.time() - start_time
print(
    f"  Running with {nthreads} threads and {g_orbits.nparticles} particles took {total_time}s"
)
g_orbits.loss_fraction(r_surface_max=r_surface_max)
g_orbits.plot_loss_fraction()
