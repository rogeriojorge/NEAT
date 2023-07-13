#!/usr/bin/env python

from neat.fields import Simple  # isort:skip
from neat.tracing import ChargedParticle, ParticleOrbit  # isort:skip
import os
import time

import numpy as np

"""                                                                           
Trace the orbit of a single particle in a
quasisymmetric stellarator                 
"""

# Initialize an alpha particle at a radius = r_initial
r_initial = 0.2  # initial normalized toroidal magnetic flux (radial VMEC coordinate)
theta_initial = np.pi / 2  # initial poloidal angle
phi_initial = 1.2  # initial poloidal angle
energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
Lambda = 0.96  # = mu * B0 / energy
vpp_sign = -1  # initial sign of the parallel velocity, +1 or -1
nsamples = 100  # IGNORED - resolution in time
tfinal = 2e-4  # seconds
wout_filename = os.path.join(os.path.dirname(__file__), "inputs", "wout_ARIESCS.nc")
B_scale = 1  # Scale the magnetic field by a factor
Aminor_scale = 1  # Scale the machine size by a factor

g_field = Simple(
    wout_filename=wout_filename, B_scale=B_scale, Aminor_scale=Aminor_scale
)
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

print("Creating B contour plot")
g_orbit.plot_orbit_contourB(show=False)

print("Creating parameter plot")
g_orbit.plot(show=False)

print("Creating 2D plot")
g_orbit.plot_orbit(show=False)

print("Creating 3D plot")
g_orbit.plot_orbit_3d(show=True)

print("Creating animation plot")
g_orbit.plot_animation(show=True)
