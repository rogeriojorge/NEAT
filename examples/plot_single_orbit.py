#!/usr/bin/env python3

import time

import numpy as np

from neat.fields import stellna_qs
from neat.tracing import charged_particle, particle_orbit

"""                                                                           
Trace the orbit of a single particle in a
quasisymmetric stellarator                 
"""

g_field = stellna_qs.from_paper(1)

r0 = 0.1
theta0 = np.pi
energy = 1e4
Lambda = 1

g_particle = charged_particle(r0=r0, theta0=theta0, energy=energy, Lambda=1)
print("Starting particle tracer")
start_time = time.time()
g_orbit = particle_orbit(g_particle, g_field, nsamples=500, Tfinal=1000)
total_time = time.time() - start_time
print(f"Finished in {total_time}s")

print("Creating 2D plot")
g_orbit.plot_orbit()

print("Creating 3D plot")
g_orbit.plot_orbit_3D()
