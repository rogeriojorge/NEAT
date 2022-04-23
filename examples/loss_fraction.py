#!/usr/bin/env python3

import matplotlib.pyplot as plt

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
g_orbits = particle_ensemble_orbit(g_particle, g_field, nthreads=nthreads)
g_orbits.calculate_loss_fraction(r_surface_max=r_surface_max)

plt.plot(g_orbits.time, g_orbits.loss_fraction)
plt.xlabel("Time")
plt.ylabel("Loss Fraction")
plt.show()
