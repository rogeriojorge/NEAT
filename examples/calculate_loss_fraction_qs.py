#!/usr/bin/env python

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
r_initial = 0.96  # meters
r_max = 0.989  # meters
B0 = 5.3267  # Tesla, magnetic field on-axis
energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
ntheta = 2  # resolution in theta
nphi = 2  # resolution in phi
nlambda_trapped = 2  # number of pitch angles for trapped particles
nlambda_passing = 2  # number of pitch angles for passing particles
nsamples = 1000  # resolution in time
tfinal = 1e-5  # seconds
nthreads_array = [4]
stellarator_index = "precise QA"
Rmajor_ARIES = 7.7495 * 2
Rminor_ARIES = 1.7044
g_field_basis = StellnaQS.from_paper(stellarator_index, B0=B0, nphi=401)
g_field = StellnaQS(
    rc=g_field_basis.rc * Rmajor_ARIES,
    zs=g_field_basis.zs * Rmajor_ARIES,
    etabar=g_field_basis.etabar / Rmajor_ARIES,
    B2c=g_field_basis.B2c * (B0 / Rmajor_ARIES / Rmajor_ARIES),
    B0=B0,
    nfp=g_field_basis.nfp,
    order="r3",
    nphi=401,
)
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
plt.show()
# print(
#     f"Final loss fraction with jacobian weights and B20 constant = {g_orbits.loss_fraction_array[-1]*100}%"
# )
# # g_orbits.plot_loss_fraction()

# g_orbits.loss_fraction(r_max=r_max, jacobian_weight=False)
# plt.semilogx(
#     g_orbits.time,
#     g_orbits.loss_fraction_array,
#     label="Without jacobian weights and B20 constant",
# )
# print(
#     f"Final loss fraction without jacobian weights and B20 constant = {g_orbits.loss_fraction_array[-1]*100}%"
# )
# g_orbits.plot_loss_fraction()
import numpy as np

for i in np.arange(0, 10, 1):
    plt.plot(g_orbits.time, g_orbits.r_pos[i], label=str(i))
plt.legend()
plt.show()

for i in np.arange(0, 5, 1):
    plt.plot(g_orbits.time, g_orbits.r_pos[i], label=str(i))
plt.legend()
plt.show()

for i in np.arange(0, 3, 1):
    plt.plot(g_orbits.time, g_orbits.r_pos[i], label=str(i))
plt.legend()
plt.show()

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

# plt.xlabel("Time (s)")
# plt.ylabel("Loss Fraction")
# plt.tight_layout()
# plt.legend()
# plt.show()
