#!/usr/bin/env python3

import os
import time

import numpy as np

from neat.fields import Vmec
from neat.tracing import ChargedParticle, ParticleOrbit

"""                                                                           
Trace the orbit of a single particle in a
vmec equilibrium                
"""

# Initialize an alpha particle at a radius = r_initial
r_initial = 0.4  # meters
theta_initial = np.pi / 2  # initial poloidal angle
phi_initial = np.pi  # initial poloidal angle
energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
Lambda = 0.97  # = mu * B0 / energy
vpp_sign = -1  # initial sign of the parallel velocity, +1 or -1
nsamples = 2000  # resolution in time
tfinal = 9.5e-5  # seconds
wout_filename = f"{os.path.join(os.path.dirname(__file__))}/inputs/wout_W7X.nc"

g_field = Vmec(wout_filename=wout_filename)
g_particle = ChargedParticle(
    r_initial=r_initial,
    theta_initial=theta_initial,
    phi_initial=phi_initial,
    energy=energy,
    Lambda=Lambda,
    charge=charge,
    mass=mass,
    vpp_sign=vpp_sign,
)
print("Starting particle tracer")
start_time = time.time()
g_orbit = ParticleOrbit(g_particle, g_field, nsamples=nsamples, tfinal=tfinal)
total_time = time.time() - start_time
print(f"Finished in {total_time}s")

print("Creating parameter plot")
g_orbit.plot(show=False)

# print("Creating 2D plot")
# g_orbit.plot_orbit(show=False)

print("Creating 3D plot")
g_orbit.plot_orbit_3d(show=True)

# print("Creating animation plot")
# g_orbit.plot_animation(show=True)
