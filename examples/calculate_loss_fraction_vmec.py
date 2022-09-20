#!/usr/bin/env python

from neat.fields import Simple  # isort:skip
from neat.tracing import ChargedParticleEnsemble, ParticleEnsembleOrbit_Simple

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
r_initial = 0.4  # initial normalized toroidal magnetic flux (radial VMEC coordinate)
energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
nparticles = 64  # number of particles
nsamples = 1000  # resolution in time, not used in this version of the code
tfinal = 1e-2  # seconds
wout_filename = f"{os.path.join(os.path.dirname(__file__))}/inputs/wout_ARIESCS.nc"
B_scale = 1  # Scale the magnetic field by a factor
Aminor_scale = 1  # Scale the machine size by a factor
# Define the min/max parallel velocities (trapped particles lie close to vparallel=0)
vparallel_over_v_min = -1.0
vparallel_over_v_max = 1.0

g_field = Simple(
    wout_filename=wout_filename, B_scale=B_scale, Aminor_scale=Aminor_scale
)
g_particle = ChargedParticleEnsemble(
    r_initial=r_initial,
    energy=energy,
    charge=charge,
    mass=mass,
)
print("Starting particle tracer")
start_time = time.time()
g_orbits = ParticleEnsembleOrbit_Simple(
    g_particle,
    g_field,
    nsamples=nsamples,
    tfinal=tfinal,
    nparticles=nparticles,
    vparallel_over_v_min=vparallel_over_v_min,
    vparallel_over_v_max=vparallel_over_v_max,
)
total_time = time.time() - start_time
print(f"  Running with {g_orbits.nparticles} particles took {total_time}s")

# Plot resulting loss fraction
g_orbits.plot_loss_fraction()
