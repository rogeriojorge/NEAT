#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from simsopt.mhd import Vmec

from neat.tracing import ChargedParticleEnsemble, ParticleEnsembleOrbit_Vmec

from neat.fields import StellnaQS, Vmec as Vmec_NEAT  # isort:skip

import os  # isort:skip
import time  # isort:skip

"""                                                                           
Calculate the loss fraction of a distribution of particles
in a VMEC equilibrium                 
"""

# Initialize an ensemble of alpha particles at a radius = r_initial
# Calculate loss fraction at a radius = r_max
# Test OpenMP parallelization with an array of threads = nthreads_array
# The total number of particles is ntheta * nphi * (nlambda_passing+nlambda_trapped) * 2 (particles with v_parallel = +1, -1)
s_initial = 0.96
r_max = 0.989
B0 = 5.3267  # Tesla, magnetic field on-axis (ARIES-CS)
energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
nsamples = 1000  # resolution in time
tfinal = 1e-5  # seconds
constant_b20 = False  # use a constant B20 (mean value) or the real function
s_boundary = 1

ntheta = 2  # resolution in theta
nphi = 2  # resolution in phi
nlambda_trapped = 2  # number of pitch angles for trapped particles
nlambda_passing = 2  # number of pitch angles for passing particles
nthreads = 4

Rmajor_ARIES = 7.7495 * 2
Rminor_ARIES = 1.7044
r_avg = Rminor_ARIES * s_boundary

filename_vmec = f"input.nearaxis_sboundary{Rmajor_ARIES/r_avg}"
wout_filename_vmec = f"wout_nearaxis_sboundary{Rmajor_ARIES/r_avg}_000_000000.nc"

g_field_basis = StellnaQS.from_paper("precise QA", B0=B0, nphi=401)
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

# g_field.to_vmec(filename=filename_vmec,r=r_avg, params={"ntor":12, "mpol":12, "niter_array":[10000,10000,20000],'ftol_array':[1e-12,1e-16,1e-17],'ns_array':[16,49,101]}, ntheta=48, ntorMax=48) #standard ntheta=20, ntorMax=14
# vmec=Vmec(filename=filename_vmec, verbose=True)
# vmec.run()

g_field_vmec = Vmec_NEAT(wout_filename=wout_filename_vmec, maximum_s=1.0)

g_particle = ChargedParticleEnsemble(
    r_initial=s_initial,
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
start_time = time.time()
g_orbits = ParticleEnsembleOrbit_Vmec(
    g_particle,
    g_field_vmec,
    tfinal=tfinal,
    nsamples=nsamples,
    nthreads=nthreads,
)

total_time = time.time() - start_time

print(f"  Running with {g_orbits.nparticles} particles took {total_time}s")
print(f"  Final loss fraction = {g_orbits.total_particles_lost}")

g_orbits.loss_fraction(r_max=r_max, jacobian_weight=True)
plt.semilogx(
    g_orbits.time,
    g_orbits.loss_fraction_array,
    label="With jacobian weights and B20 constant",
)
plt.show()

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
# for i in np.arange(0,10,1):
#     plt.plot(g_orbits.time,g_orbits.r_pos[i],label=str(i))
# plt.legend()
# plt.show()

# for i in np.arange(0,5,1):
#     plt.plot(g_orbits.time,g_orbits.r_pos[i],label=str(i))
# plt.legend()
# plt.show()

# for i in np.arange(0,3,1):
#     plt.plot(g_orbits.time,g_orbits.r_pos[i],label=str(i))
# plt.legend()
# plt.show()
