#!/usr/bin/env python3

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
s_initial = 0.4  # initial normalized toroidal magnetic flux (radial VMEC coordinate)
energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
nparticles = 64  # number of particles
tfinal = 1e-2  # seconds
wout_filename = os.path.join(os.path.dirname(__file__), "inputs", "wout_ARIESCS.nc")
B_scale = 1  # Scale the magnetic field by a factor
Aminor_scale = 1  # Scale the machine size by a factor

g_field = Simple(
    wout_filename=wout_filename, B_scale=B_scale, Aminor_scale=Aminor_scale
)
g_particle = ChargedParticleEnsemble(
    r_initial=s_initial,
    energy=energy,
    charge=charge,
    mass=mass,
)
print("Starting particle tracer")
start_time = time.time()
g_orbits = ParticleEnsembleOrbit_Simple(
    g_particle,
    g_field,
    tfinal=tfinal,
    nparticles=nparticles,
)
total_time = time.time() - start_time
print(f"  Running with {g_orbits.nparticles} particles took {total_time}s")
print(
    f"  Loss fraction = {100*g_orbits.loss_fraction_array[-1]}% for a time of {tfinal}s"
)

# Plot resulting loss fraction
g_orbits.plot_loss_fraction()
