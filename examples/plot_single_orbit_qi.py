#!/usr/bin/env python3

import time

import numpy as np

from neat.fields import stellna
from neat.tracing import charged_particle, particle_orbit

"""                                                                           
Trace the orbit of a single particle in a
quasisymmetric stellarator                 
"""

# Initialize an alpha particle at a radius = r_initial
r_initial = 0.1  # meters
theta0 = np.pi / 2  # initial poloidal angle
phi0 = np.pi  # initial poloidal angle
B0 = 5  # Tesla, mean magnetic field on-axis
energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
Lambda = 0.98  # = mu * B0 / energy
vpp_sign = -1  # initial sign of the parallel velocity, +1 or -1
nsamples = 1000  # resolution in time
Tfinal = 6e-5  # seconds
B20_constant = False  # use a constant B20 (mean value) or the real function

# Quasi-isodynamic
g_field_temp = stellna.from_paper("QI NFP1 r2", nphi=201)
g_field = stellna.from_paper(
    "QI NFP1 r2", B0_vals=np.array(g_field_temp.B0_vals) * B0, nphi=201
)
# Quasi-axisymmetric
# g_field = stellna.from_paper(2, B0=B0, nphi=201)
# Quasi-helically symmetric
# g_field = stellna.from_paper(4, B0=B0, nphi=201)
g_particle = charged_particle(
    r0=r_initial,
    theta0=theta0,
    phi0=phi0,
    energy=energy,
    Lambda=Lambda,
    charge=charge,
    mass=mass,
    vpp_sign=vpp_sign,
)
print("Starting particle tracer")
start_time = time.time()
g_orbit = particle_orbit(
    g_particle, g_field, nsamples=nsamples, Tfinal=Tfinal, B20_constant=B20_constant
)
total_time = time.time() - start_time
print(f"Finished in {total_time}s")

print("Creating parameter plot")
g_orbit.plot(show=False)

# print("Creating 2D plot")
# g_orbit.plot_orbit(show=False)

print("Creating 3D plot")
g_orbit.plot_orbit_3D(r_surface=0.1, show=False)

print("Creating animation plot")
g_orbit.plot_animation(r_surface=0.1, distance=8, show=True)