#!/usr/bin/env python3

import os
import time

import numpy as np
from pyrsistent import v
import vmec
import logging
from mpi4py import MPI
from neat.fields import StellnaQS, Vmec
from neat.tracing import ChargedParticle, ParticleOrbit
import matplotlib.pyplot as plt
import subprocess

"""                                                                           
Trace the orbit of a single particle in a
quasisymmetric stellarator                 
"""

# Initialize an alpha particle at a radius = r_initial
r_initial = 0.1  # meters
theta_initial = np.pi / 2  # initial poloidal angle
phi_initial = np.pi  # initial poloidal angle
B0 = 5  # Tesla, magnetic field on-axis
energy = 3.52e3  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
Lambda = 0.2  # = mu * B0 / energy
vpp_sign = -1  # initial sign of the parallel velocity, +1 or -1
nsamples = 1000  # resolution in time
tfinal = 6e-5  # seconds
constant_b20 = True  # use a constant B20 (mean value) or the real function
#wout_filename = f"{os.path.join(os.path.dirname(__file__))}/inputs/wout_W7X.nc"
filename = "input.nearaxis"
wout_filename = "wout_nearaxis.nc"

g_field = StellnaQS.from_paper(1, B0=B0)
#g_field.to_vmec(filename=filename)
#subprocess.run([f"{(os.path.dirname(__file__))}./xvmec2000", filename])
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

#Notes on ploting
#Show=True means all before are plotted
#Save movie doesnt seem to work
#Orbit lines in vmec are way closer to middle

#print("Creating parameter plot")
#g_orbit.plot(show=False)

# print("Creating 2D plot")
# g_orbit.plot_orbit(show=False)

#print("Creating 3D plot")
#g_orbit.plot_orbit_3d(show=True)

#print("Creating animation plot")
#g_orbit.plot_animation(show=False)

#print("Creating parameter plot 2")
#g_orbit_vmec.plot(show=False)

#print("Creating 3D plot 2")
#g_orbit_vmec.plot_orbit_3d(show=True)

#print("Creating animation plot 2")
#g_orbit_vmec.plot_animation(show=False)

print("Calculating differences between near axis and vmec")
diff_r=(g_orbit.r_pos - g_orbit_vmec.r_pos)/g_orbit.r_pos
diff_theta=(g_orbit.theta_pos - g_orbit_vmec.theta_pos)/(2*np.pi)
diff_phi=(g_orbit.varphi_pos - g_orbit_vmec.varphi_pos)/(2*np.pi)

#ax=plt.axes()
_ = plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 1)
plt.plot(g_orbit.time, diff_r, label='Particle 1')
plt.xlabel('t')
plt.ylabel('diff_r')
plt.subplot(1, 3, 2)
plt.plot(g_orbit.time, diff_theta, label='Particle 1')
plt.xlabel('t')
plt.ylabel('diff_theta')
plt.subplot(1, 3, 3)
plt.plot(g_orbit.time, diff_phi, label='Particle 1')
plt.xlabel('t')
plt.ylabel('diff_phi')
plt.title('First particle tracing')
plt.legend()
plt.show()

# For when we know how to runvmec

#ictrl = np.zeros(5, dtype=np.int32)
#ictrl[:] = 0
#ictrl[0] = 1 + 2 + 4 + 8
#logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)
#logging.basicConfig(level=logging.INFO)
#fcomm = MPI.COMM_WORLD.py2f()
#logger.info("Calling runvmec. ictrl={} comm={}".format(ictrl, fcomm))
#vmec.runvmec(ictrl, filename, True, fcomm, '') 