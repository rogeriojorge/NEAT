#!/usr/bin/env python

import time

import matplotlib.pyplot as plt
import numpy as np
from simsopt.mhd import Vmec

from neat.fields import Simple, StellnaQS
from neat.tracing import (
    ChargedParticleEnsemble,
    ParticleEnsembleOrbit,
    ParticleEnsembleOrbit_Simple,
)

"""                                                                           
Calculate the loss fraction of a distribution of particles
in a quasisymmetric stellarator                   
"""

# Initialize an ensemble of alpha particles at a radius = r_initial
# Calculate loss fraction at a radius = r_max
# Test OpenMP parallelization with an array of threads = nthreads_array
# The total number of particles is ntheta * nphi * (nlambda_passing+nlambda_trapped) * 2 (particles with v_parallel = +1, -1)

s_boundary = 1.0
s_initial = 0.3
r_initial = 0.06  # meters
r_max = 0.1  # meters
B0 = 5  # Tesla, magnetic field on-axis
energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
ntheta = 2  # resolution in theta
nphi = 2  # resolution in phi
nlambda_trapped = 18  # number of pitch angles for trapped particles
nlambda_passing = 3  # number of pitch angles for passing particles
nsamples = 100  # resolution in time
tfinal = 1e-6  # seconds
nthreads_array = [6]
stellarator = "precise QA"
filename = f"input.nearaxis_sboundary{s_boundary}"
wout_filename = f"wout_nearaxis_sboundary{s_boundary}_000_000000.nc"

Rmajor_ARIES = 7.7495
Rminor_ARIES = 1.7044
rminor_factor = 1.4
r_min = 0.35
r_avg = Rminor_ARIES / rminor_factor
nparticles = nlambda_passing + nlambda_trapped
notrace_passing = 0  # If 1 skip tracing of passing particles

r_initial = r_avg * np.sqrt(s_boundary) * np.sqrt(s_initial)  # meters
r_max = Rminor_ARIES  # meters
Aminor_scale = Rmajor_ARIES / Rminor_ARIES

g_field_basis = StellnaQS.from_paper(stellarator, B0=B0, nphi=131)
g_field = StellnaQS(
    rc=g_field_basis.rc * Rmajor_ARIES,
    zs=g_field_basis.zs * Rmajor_ARIES,
    etabar=g_field_basis.etabar / Rmajor_ARIES,
    B2c=g_field_basis.B2c * (B0 / Rmajor_ARIES / Rmajor_ARIES),
    B0=B0,
    nfp=g_field_basis.nfp,
    order="r3",
    nphi=131,
)

g_field.to_vmec(filename=filename, r=r_avg * np.sqrt(s_boundary))

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
    label="With jacobian weights and B20 constant",
)

print(
    f"Final loss fraction with jacobian weights and B20 constant = {g_orbits.loss_fraction_array[-1]*100}%"
)
g_orbits.plot_loss_fraction()

# g_orbits.loss_fraction(r_max=r_max, jacobian_weight=False)
# plt.semilogx(g_orbits.time, g_orbits.loss_fraction_array, label="Without jacobian weights and B20 constant")

# print(f"Final loss fraction without jacobian weights and B20 constant = {g_orbits.loss_fraction_array[-1]*100}%")
# g_orbits.plot_loss_fraction()

vmec = Vmec(filename=filename, verbose=False)
vmec.run()
g_field_simple = Simple(
    wout_filename=wout_filename, B_scale=B0, Aminor_scale=Aminor_scale
)

g_particle_simple = ChargedParticleEnsemble(
    r_initial=s_initial, energy=energy, charge=charge, mass=mass
)

print("Starting particle tracer")
start_time = time.time()

g_orbits_simple = ParticleEnsembleOrbit_Simple(
    g_particle_simple,
    g_field_simple,
    tfinal=tfinal,
    nthreads=nthreads_array[0],
    nparticles=nparticles,
    nsamples=nsamples,
    notrace_passing=notrace_passing,
)

total_time = time.time() - start_time
print(f"  Running with {g_orbits_simple.nparticles} particles took {total_time}s")
print(f"  Final loss fraction = {g_orbits_simple.total_particles_lost}")

# g_orbits_simple.loss_fraction(r_max=r_max, jacobian_weight=True)
# plt.semilogx( g_orbits_simple.time, g_orbits_simple.loss_fraction_array)
# Plot resulting loss fraction
g_orbits_simple.plot_loss_fraction()


# print("Starting particle tracer with B20 not constant")
# constant_b20 = False
# threads_vs_time = []
# for nthreads in nthreads_array:
#     start_time = time.time()
#     g_orbits = ParticleEnsembleOrbit(
#         g_particle,
#         g_field,
#         nsamples=nsamples,
#         tfinal=tfinal,
#         nthreads=nthreads,
#         constant_b20=constant_b20,
#     )
#     total_time = time.time() - start_time
#     print(
#         f"  Running with {nthreads} threads and {g_orbits.nparticles} particles took {total_time}s"
#     )
#     threads_vs_time.append([nthreads, total_time])
# g_orbits.loss_fraction(r_max=r_max, jacobian_weight=True)
# plt.semilogx(
#     g_orbits.time,
#     g_orbits.loss_fraction_array,
#     label="With jacobian weights and B20 not constant",
# )
# print(
#     f"Final loss fraction with jacobian weights and B20 not constant = {g_orbits.loss_fraction_array[-1]*100}%"
# )
# # g_orbits.plot_loss_fraction()

# g_orbits.loss_fraction(r_max=r_max, jacobian_weight=False)
# plt.semilogx(
#     g_orbits.time,
#     g_orbits.loss_fraction_array,
#     label="Without jacobian weights and B20 not constant",
# )
# print(
#     f"Final loss fraction without jacobian weights and B20 not constant = {g_orbits.loss_fraction_array[-1]*100}%"
# )
# # g_orbits.plot_loss_fraction()

plt.xlabel("Time (s)")
plt.ylabel("Loss Fraction")
plt.tight_layout()
plt.legend()
plt.show()
