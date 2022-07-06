#!/usr/bin/env python3

import logging
import os
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
from sympy import I
import vmec
from mpi4py import MPI

from neat.fields import StellnaQS, Vmec
from neat.tracing import ChargedParticle, ParticleOrbit

"""                                                                           
Trace the orbit of a single particle in a
quasisymmetric stellarator using Near Axis and VMEC                 
"""

# Initialize an alpha particle at a radius = r_initial
s_initial = 0.05  # psi/psi_a
theta_initial = 0   # initial poloidal angle
phi_initial = 0  # initial toroidal angle
B0 = 1  # Tesla, magnetic field on-axis
energy = 3.52e4  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
Lambda = 0.9  # = mu * B0 / energy
vpp_sign = -1  # initial sign of the parallel velocity, +1 or -1
nsamples = 2000 # resolution in time
tfinal = 1e-4  # seconds
constant_b20 = False  # use a constant B20 (mean value) or the real function
filename = "input.nearaxis"
wout_filename = "wout_nearaxis.nc"

g_field = StellnaQS.from_paper(1, B0=B0, nphi=201)
# g_field.to_vmec(filename=filename)
#subprocess.run([f"{os.path.join(os.path.dirname(__file__))}./xvmec2000", filename])
g_field_vmec = Vmec(wout_filename=wout_filename)

psi_a=(B0*0.1)*(0.1)/2
g_particle = ChargedParticle(
    r_initial=0.1*np.sqrt(s_initial),#np.sqrt(2*r_initial*psi_a/B0),
    theta_initial=theta_initial, #doesnt affect phi_cil
    phi_initial=phi_initial,     #affects phi_cil
    energy=energy,
    Lambda=Lambda,
    charge=charge,
    mass=mass,
    vpp_sign=vpp_sign,
)

g_particle_vmec = ChargedParticle(
    r_initial=s_initial,
    theta_initial=theta_initial,     #affects phi_cil
    phi_initial=phi_initial,     #doesnt affect phi_cil
    energy=energy,
    Lambda=Lambda,
    charge=charge,
    mass=mass,
    vpp_sign=-vpp_sign,
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
    g_particle_vmec,
    g_field_vmec,
    nsamples=nsamples,
    tfinal=tfinal,
    constant_b20=constant_b20,
)
total_time_vmec = time.time() - start_time_vmec
print(f"Finished in {total_time_vmec}s")


print("Creating parameter plot")
g_orbit.plot(show=False)

# print("Creating 2D plot")
# g_orbit.plot_orbit(show=False)

print("Creating 3D plot")
g_orbit.plot_orbit_3d(show=False)

print("Creating parameter plot")
g_orbit_vmec.plot(show=False)

# print("Creating 2D plot")
# g_orbit.plot_orbit(show=False)

print("Creating 3D plot")
g_orbit_vmec.plot_orbit_3d(show=False)
"""""
print("Creating animation plot")
g_orbit.plot_animation(show=True)

"""
"""
print("Creating parameter plot 2")
g_orbit_vmec.plot(show=False)

print("Creating 3D plot 2")
g_orbit_vmec.plot_orbit_3d(show=False)

print("Creating animation plot 2")
g_orbit_vmec.plot_animation(show=True)

"""

# Notes on ploting
# Show=True means all before are plotted
# Save movie doesnt seem to work
# Orbit lines in vmec are way closer to middle
# Interpolation of VMEC at E propto 3e5 fails for l>0.8
# Phi and theta seem to be switched
# Cant seem to achieve 3e6, only 3e5, for B=5 -> Problem solved with B=10
# phi=0 and theta=0 vmec gets a uniform path 

print("Calculating differences between near axis and vmec")
"""
diff_r=np.zeros(len(g_orbit.time))
diff_Z=np.zeros(len(g_orbit.time))
for i in range(len(g_orbit.time)):
    diff_r[i] = (
        g_orbit.rpos_cylindrical[0][i] - g_orbit_vmec.rpos_cylindrical[0][i]
    ) / g_orbit_vmec.rpos_cylindrical[0][i]
    diff_Z[i] = (g_orbit.rpos_cylindrical[1][i] - g_orbit_vmec.rpos_cylindrical[1][i]
    ) / (g_orbit_vmec.rpos_cylindrical[1][i])
    if diff_Z[i]>1:
        print(g_orbit_vmec.rpos_cylindrical[1][i])
"""
diff_r = (
        g_orbit.rpos_cylindrical[0] - g_orbit_vmec.rpos_cylindrical[0]
    ) / g_orbit_vmec.rpos_cylindrical[0]
diff_Z = (g_orbit.rpos_cylindrical[1] - g_orbit_vmec.rpos_cylindrical[1]
    ) / (np.max(abs(g_orbit_vmec.rpos_cylindrical[1])))
diff_phi= (
    np.unwrap(np.mod(g_orbit.rpos_cylindrical[2], 2*np.pi)) - np.unwrap(np.mod(g_orbit_vmec.rpos_cylindrical[2], 2*np.pi))
) / (2 * np.pi)

_ = plt.figure(figsize=(20, 8))
plt.subplot(3, 4, 1)
plt.plot(g_orbit.time*1e6, g_orbit.rpos_cylindrical[0], label='Particle 1')
plt.xlabel(r't ($\mu$s)')
plt.ylabel(r'$R$')
plt.subplot(3, 4, 2)
plt.plot(g_orbit.time*1e6, g_orbit.rpos_cylindrical[1], label='Particle 1')
plt.xlabel(r't  ($\mu$s)')
plt.ylabel(r'$Z$')
plt.subplot(3, 4, 3)
plt.plot(g_orbit.time*1e6, np.mod(g_orbit.rpos_cylindrical[2], 2*np.pi), label='Particle 1')
plt.xlabel(r't  ($\mu$s)')
plt.ylabel(r'$\Phi$')
plt.subplot(3, 4, 4)
plt.plot(g_orbit.rpos_cylindrical[0], g_orbit.rpos_cylindrical[1], label='Particle 1')
plt.xlabel(r'$R$')
plt.ylabel(r'$Z$')
plt.subplot(3, 4, 5)
plt.plot(g_orbit_vmec.time*1e6, g_orbit_vmec.rpos_cylindrical[0], label='Particle 1')
plt.xlabel(r't ($\mu$s)')
plt.ylabel(r'$R_V$')
plt.subplot(3, 4, 6)
plt.plot(g_orbit_vmec.time*1e6, g_orbit_vmec.rpos_cylindrical[1], label='Particle 1')
plt.xlabel(r't  ($\mu$s)')
plt.ylabel(r'$Z_V$')
plt.subplot(3, 4, 7)
plt.plot(g_orbit_vmec.time*1e6, np.mod(g_orbit_vmec.rpos_cylindrical[2], 2*np.pi), label='Particle 1')
plt.xlabel(r't  ($\mu$s)')
plt.ylabel(r'$\Phi_V $')
plt.subplot(3, 4, 8)
plt.plot(g_orbit_vmec.rpos_cylindrical[0], g_orbit_vmec.rpos_cylindrical[1], label='Particle 1')
plt.xlabel(r'$R_V$')
plt.ylabel(r'$Z_V$')
plt.subplot(3, 4, 9)
plt.plot(g_orbit_vmec.time*1e6, diff_r*100, label='Particle 1')
plt.xlabel(r't ($\mu$s)')
plt.ylabel(r'$\Delta  R (\%)$')
plt.subplot(3, 4, 10)
plt.plot(g_orbit_vmec.time*1e6, diff_Z, label='Particle 1')
plt.xlabel(r't  ($\mu$s)')
plt.ylabel(r'$\Delta  Z_{normalized}$')
plt.subplot(3, 4, 11)
plt.plot(g_orbit_vmec.time*1e6, diff_phi, label='Particle 1')
plt.xlabel(r't  ($\mu$s)')
plt.ylabel(r'$\Delta \Phi (turns)$')
#plt.subplot(3, 4, 12)
#plt.plot(diff_r,diff_Z, label='Particle 1')
#plt.xlabel(r'$\Delta R$')
#plt.ylabel(r'$\Delta Z$')
#plt.title('First particle tracing')
plt.legend()
plt.show()

# For when we know how to runvmec -> insert after g_field.to_vmec(filename=filename)

# ictrl = np.zeros(5, dtype=np.int32)
# ictrl[:] = 0
# ictrl[0] = 1 + 2 + 4 + 8
# logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)
# logging.basicConfig(level=logging.INFO)
# fcomm = MPI.COMM_WORLD.py2f()
# logger.info("Calling runvmec. ictrl={} comm={}".format(ictrl, fcomm))
# vmec.runvmec(ictrl, filename, True, fcomm, '')
