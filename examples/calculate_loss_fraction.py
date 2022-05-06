#!/usr/bin/env python3

import time

from neat.fields import stellna_qs
from neat.tracing import charged_particle_ensemble, particle_ensemble_orbit

"""                                                                           
Calculate the loss fraction of a distribution of particles
in a quasisymmetric stellarator                   
"""

# Initialize an ensemble of alpha particles at a radius = r_initial
# Calculate loss fraction at a radius = r_max
# Test OpenMP parallelization with an array of threads = nthreads_array
# The total number of particles is ntheta * nphi * (nlambda_passing+nlambda_trapped) * 2 (particles with v_parallel = +1, -1)
r_initial = 0.05  # meters
r_max = 0.1  # meters
B0 = 4  # Tesla, magnetic field on-axis
energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
ntheta = 14  # resolution in theta
nphi = 10  # resolution in phi
nlambda_trapped = 20  # number of pitch angles for trapped particles
nlambda_passing = 3  # number of pitch angles for passing particles
nsamples = 1000  # resolution in time
Tfinal = 1e-4  # seconds
nthreads_array = [1, 2, 4, 8]

g_field = stellna_qs.from_paper(2, B0=B0, etabar=0.1)
g_particle = charged_particle_ensemble(
    r0=r_initial,
    r_max=r_max,
    energy=energy,
    charge=charge,
    mass=mass,
    ntheta=ntheta,
    nphi=nphi,
    nlambda_trapped=nlambda_trapped,
    nlambda_passing=nlambda_passing,
)
print("Starting particle tracer")
threads_vs_time = []
for nthreads in nthreads_array:
    start_time = time.time()
    g_orbits = particle_ensemble_orbit(
        g_particle, g_field, nsamples=nsamples, Tfinal=Tfinal, nthreads=nthreads
    )
    total_time = time.time() - start_time
    print(
        f"  Running with {nthreads} threads and {g_orbits.nparticles} particles took {total_time}s"
    )
    threads_vs_time.append([nthreads, total_time])
g_orbits.loss_fraction(r_max=r_max)
print(f"Final loss fraction = {g_orbits.loss_fraction_array[-1]*100}%")
g_orbits.plot_loss_fraction()
