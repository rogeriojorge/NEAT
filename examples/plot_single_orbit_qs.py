#!/usr/bin/env python3

import os
import time

import numpy as np
import vmec
from mpi4py import MPI
from neat.fields import StellnaQS, Vmec
from neat.tracing import ChargedParticle, ParticleOrbit

"""                                                                           
Trace the orbit of a single particle in a
quasisymmetric stellarator                 
"""

# Initialize an alpha particle at a radius = r_initial
r_initial = 0.1  # meters
theta_initial = np.pi / 2  # initial poloidal angle
phi_initial = np.pi  # initial poloidal angle
B0 = 5  # Tesla, magnetic field on-axis
energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
Lambda = 0.98  # = mu * B0 / energy
vpp_sign = -1  # initial sign of the parallel velocity, +1 or -1
nsamples = 1000  # resolution in time
tfinal = 6e-5  # seconds
constant_b20 = True  # use a constant B20 (mean value) or the real function
#wout_filename = f"{os.path.join(os.path.dirname(__file__))}/inputs/wout_W7X.nc"
filename = "input.nearaxis"
wout_filename = "output_nearaxis.nc"

g_field = StellnaQS.from_paper(1, B0=B0)
g_field.to_vmec(filename=filename)
ictrl = np.zeros(5, dtype=np.int32)
ictrl[:] = 0
ictrl[0] = 1 + 2 + 4 + 8
vmec.runvmec(ictrl, filename, True, MPI.COMM_WORLD.py2f(), '') 

g_field_vmec = Vmec(wout_filename=wout_filename)

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
print("Starting particle tracer 1")
start_time = time.time()
g_orbit = ParticleOrbit(
    g_particle, g_field, nsamples=nsamples, tfinal=tfinal, constant_b20=constant_b20
)
total_time = time.time() - start_time
print(f"Finished in {total_time}s")

print("Starting particle tracer 2")
start_time_vmec = time.time()
g_orbit_vmec = ParticleOrbit(
    g_particle, g_field_vmec, nsamples=nsamples, tfinal=tfinal, constant_b20=constant_b20
)
total_time_vmec = time.time() - start_time_vmec
print(f"Finished in {total_time_vmec}s")

print("Creating parameter plot")
g_orbit.plot(show=False)

# print("Creating 2D plot")
# g_orbit.plot_orbit(show=False)

print("Creating 3D plot")
g_orbit.plot_orbit_3d(show=False)

print("Creating animation plot")
g_orbit.plot_animation(show=True)

#print("Creating parameter plot 2")
#g_orbit_vmec.plot(show=False)

#print("Creating 3D plot 2")
#g_orbit_vmec.plot_orbit_3d(show=False)

#print("Creating animation plot 2")
#g_orbit_vmec.plot_animation(show=True)
