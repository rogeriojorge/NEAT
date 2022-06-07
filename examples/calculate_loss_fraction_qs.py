#!/usr/bin/env python3

import time

import matplotlib.pyplot as plt

from neat.fields import StellnaQS
from neat.tracing import ChargedParticleEnsemble, ParticleEnsembleOrbit

"""                                                                           
Calculate the loss fraction of a distribution of particles
in a quasisymmetric stellarator                   
"""

# Initialize an ensemble of alpha particles at a radius = r_initial
# Calculate loss fraction at a radius = r_max
# Test OpenMP parallelization with an array of threads = nthreads_array
# The total number of particles is ntheta * nphi * (nlambda_passing+nlambda_trapped) * 2 (particles with v_parallel = +1, -1)
r_initial = 0.03  # meters
r_max = 0.08  # meters
B0 = 4  # Tesla, magnetic field on-axis
energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
ntheta = 16  # resolution in theta
nphi = 8  # resolution in phi
nlambda_trapped = 16  # number of pitch angles for trapped particles
nlambda_passing = 4  # number of pitch angles for passing particles
nsamples = 1000  # resolution in time
tfinal = 1e-4  # seconds
nthreads_array = [1, 2, 4]
stellarator_index = 1

g_field = StellnaQS.from_paper(stellarator_index, B0=B0, nphi=131)
g_particle = ChargedParticleEnsemble(
    r_initial=r_initial,
    r_max=r_max,
    energy=energy,
    charge=charge,
    mass=mass,
    ntheta=ntheta,
    nphi=nphi,
    nlambda_trapped=nlambda_trapped,
    nlambda_passing=nlambda_passing,
)
print("Starting particle tracer with B20 constant")
constant_b20 = True
threads_vs_time = []
for nthreads in nthreads_array:
    start_time = time.time()
    g_orbits = ParticleEnsembleOrbit(
        g_particle,
        g_field,
        nsamples=nsamples,
        tfinal=tfinal,
        nthreads=nthreads,
        constant_b20=constant_b20,
    )
    total_time = time.time() - start_time
    print(
        f"  Running with {nthreads} threads and {g_orbits.nparticles} particles took {total_time}s"
    )
    threads_vs_time.append([nthreads, total_time])
g_orbits.loss_fraction(r_max=r_max, jacobian_weight=True)
plt.semilogx(
    g_orbits.time,
    g_orbits.loss_fraction_array,
    label="With jacobian weights and B20 constant",
)
print(
    f"Final loss fraction with jacobian weights and B20 constant = {g_orbits.loss_fraction_array[-1]*100}%"
)
# g_orbits.plot_loss_fraction()

g_orbits.loss_fraction(r_max=r_max, jacobian_weight=False)
plt.semilogx(
    g_orbits.time,
    g_orbits.loss_fraction_array,
    label="Without jacobian weights and B20 constant",
)
print(
    f"Final loss fraction without jacobian weights and B20 constant = {g_orbits.loss_fraction_array[-1]*100}%"
)
# g_orbits.plot_loss_fraction()

print("Starting particle tracer with B20 not constant")
constant_b20 = False
threads_vs_time = []
for nthreads in nthreads_array:
    start_time = time.time()
    g_orbits = ParticleEnsembleOrbit(
        g_particle,
        g_field,
        nsamples=nsamples,
        tfinal=tfinal,
        nthreads=nthreads,
        constant_b20=constant_b20,
    )
    total_time = time.time() - start_time
    print(
        f"  Running with {nthreads} threads and {g_orbits.nparticles} particles took {total_time}s"
    )
    threads_vs_time.append([nthreads, total_time])
g_orbits.loss_fraction(r_max=r_max, jacobian_weight=True)
plt.semilogx(
    g_orbits.time,
    g_orbits.loss_fraction_array,
    label="With jacobian weights and B20 not constant",
)
print(
    f"Final loss fraction with jacobian weights and B20 not constant = {g_orbits.loss_fraction_array[-1]*100}%"
)
# g_orbits.plot_loss_fraction()

g_orbits.loss_fraction(r_max=r_max, jacobian_weight=False)
plt.semilogx(
    g_orbits.time,
    g_orbits.loss_fraction_array,
    label="Without jacobian weights and B20 not constant",
)
print(
    f"Final loss fraction without jacobian weights and B20 not constant = {g_orbits.loss_fraction_array[-1]*100}%"
)
# g_orbits.plot_loss_fraction()

plt.xlabel("Time (s)")
plt.ylabel("Loss Fraction")
plt.tight_layout()
plt.legend()
plt.show()
