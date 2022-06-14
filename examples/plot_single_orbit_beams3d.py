#!/usr/bin/env python3
import os
import time

import numpy as np

from neat.fields import Vmec
from neat.tracing import ChargedParticle, ParticleOrbitBeams3D

"""                                                                           
Trace the orbit of a single particle in a
vmec equilibrium using BEAMS3D
"""

beams3d_executable = "/Users/rogeriojorge/local/STELLOPT/BEAMS3D/Release/xbeams3d"
results_folder = f"{os.path.join(os.path.dirname(__file__))}/inputs"
wout_filename = f"{results_folder}/wout_W7X.nc"

r_initial = 0.2  # initial normalized radial coordinate
theta_initial = np.pi / 2  # initial poloidal angle
phi_initial = 0.0  # initial toroidal angle
energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
Lambda = 0.9  # = mu * B0 / energy
vpp_sign = -1  # initial sign of the parallel velocity, +1 or -1
nsamples = 2000  # resolution in time
tfinal = 4e-5  # seconds
nr = 128  # BEAMS3D radial resolution
nz = 128  # BEAMS3D vertical resolution
nphi = 128  # BEAMS3D toroidal resolution

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
g_orbit = ParticleOrbitBeams3D(
    field=g_field,
    results_folder=results_folder,
    particle=g_particle,
    nsamples=nsamples,
    tfinal=tfinal,
    NR=nr,
    NZ=nz,
    NPHI=nphi,
)
g_orbit.run(beams3d_executable=beams3d_executable)
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
