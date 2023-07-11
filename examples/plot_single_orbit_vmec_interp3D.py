#!/usr/bin/env python

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
r_initial = 0.2  # initial normalized toroidal magnetic flux (radial VMEC coordinate)
theta_initial = 0.1  # initial poloidal angle
phi_initial = 0.1  # initial poloidal angle
energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
Lambda = 0.2  # = mu * B0 / energy
vpp_sign = 1  # initial sign of the parallel velocity, +1 or -1
nsamples_array = [1000]  # resolution in time
tfinal = 5e-5  # seconds
wout_filename = os.path.join(os.path.dirname(__file__), "inputs", "wout_ARIESCS.nc")
ns = 10  # radial interpolation resolution
ntheta = 10  # poloidal interpolation resolution
nzeta = 10  # toroidal interpolation resolution

g_field_interp = Vmec(
    wout_filename=wout_filename, interp3D=True, ns=ns, ntheta=ntheta, nzeta=nzeta
)
g_field_noninterp = Vmec(wout_filename=wout_filename, interp3D=False)

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

time_interp = []
time_noninterp = []
for nsamples in nsamples_array:
    print("=" * 80)
    print(f"nsamples = {nsamples}")
    print("  Starting particle tracer noninterp")
    start_time = time.time()
    g_orbit_noninterp = ParticleOrbit(
        g_particle, g_field_noninterp, nsamples=nsamples, tfinal=tfinal
    )
    total_time = time.time() - start_time
    print(f"  Finished noninterp in {total_time}s")
    time_noninterp.append(total_time)

    print("  Starting particle tracer interp")
    start_time = time.time()
    g_orbit_interp = ParticleOrbit(
        g_particle, g_field_interp, nsamples=nsamples, tfinal=tfinal
    )
    total_time = time.time() - start_time
    print(f"  Finished interp in {total_time}s")
    time_interp.append(total_time)


# print("Creating B contour plot")
# g_orbit.plot_orbit_contourB(show=False)

# print("Creating parameter plot")
# g_orbit_interp.plot(show=False)
# g_orbit_noninterp.plot(show=True)

# print("Creating 2D plot")
# g_orbit.plot_orbit(show=False)

# print("Creating 3D plot")
# g_orbit.plot_orbit_3d(show=True)

# print("Creating animation plot")
# g_orbit.plot_animation(show=True)

import matplotlib.pyplot as plt

if len(nsamples_array) > 1:
    plt.figure(figsize=(10, 6))
    plt.plot(nsamples_array, time_interp, label="interp")
    plt.plot(nsamples_array, time_noninterp, label="noninterp")
    plt.legend()
    plt.xlabel("nsamples")
    plt.ylabel("time (s)")
    plt.show()

plt.figure(figsize=(10, 6))
plt.subplot(3, 3, 1)
plt.plot(g_orbit_interp.time, g_orbit_interp.r_pos, label="interp")
plt.plot(g_orbit_noninterp.time, g_orbit_noninterp.r_pos, label="noninterp")
plt.legend()
plt.xlabel(r"$t (s)$")
plt.ylabel(r"$r$")
plt.subplot(3, 3, 2)
plt.plot(g_orbit_interp.time, g_orbit_interp.theta_pos, label="interp")
plt.plot(g_orbit_noninterp.time, g_orbit_noninterp.theta_pos, label="noninterp")
plt.legend()
plt.xlabel(r"$t (s)$")
plt.ylabel(r"$\theta$")
plt.subplot(3, 3, 3)
plt.plot(g_orbit_interp.time, g_orbit_interp.varphi_pos, label="interp")
plt.plot(g_orbit_noninterp.time, g_orbit_noninterp.varphi_pos, label="interp")
plt.legend()
plt.xlabel(r"$t (s)$")
plt.ylabel(r"$\varphi$")
plt.subplot(3, 3, 4)
plt.plot(g_orbit_interp.time, g_orbit_interp.v_parallel, label="interp")
plt.plot(g_orbit_noninterp.time, g_orbit_noninterp.v_parallel, label="interp")
plt.legend()
plt.xlabel(r"$t (s)$")
plt.ylabel(r"$v_\parallel$")
plt.subplot(3, 3, 5)
plt.plot(
    g_orbit_interp.time,
    (g_orbit_interp.total_energy - g_orbit_interp.total_energy[0])
    / g_orbit_interp.total_energy[0],
    label="interp",
)
plt.plot(
    g_orbit_noninterp.time,
    (g_orbit_noninterp.total_energy - g_orbit_noninterp.total_energy[0])
    / g_orbit_noninterp.total_energy[0],
    label="noninterp",
)
plt.legend()
plt.xlabel(r"$t (s)$")
plt.ylabel(r"$(E-E_0)/E_0$")
plt.subplot(3, 3, 6)
plt.plot(
    g_orbit_interp.rpos_cylindrical[0],
    g_orbit_interp.rpos_cylindrical[1],
    label="interp",
)
plt.plot(
    g_orbit_noninterp.rpos_cylindrical[0],
    g_orbit_noninterp.rpos_cylindrical[1],
    label="noninterp",
)
plt.legend()
plt.xlabel(r"$R$")
plt.ylabel(r"$Z$")
plt.subplot(3, 3, 7)
plt.plot(g_orbit_interp.time, g_orbit_interp.rdot, label=r"$\dot r$ interp")
plt.plot(g_orbit_noninterp.time, g_orbit_noninterp.rdot, label=r"$\dot r$ noninterp")
plt.plot(g_orbit_interp.time, g_orbit_interp.thetadot, label=r"$\dot \theta$ interp")
plt.plot(
    g_orbit_noninterp.time, g_orbit_noninterp.thetadot, label=r"$\dot \theta$ noninterp"
)
plt.plot(g_orbit_interp.time, g_orbit_interp.varphidot, label=r"$\dot \varphi$ interp")
plt.plot(
    g_orbit_noninterp.time,
    g_orbit_noninterp.varphidot,
    label=r"$\dot \varphi$ noninterp",
)
plt.xlabel(r"$t (s)$")
plt.legend()
plt.subplot(3, 3, 8)
plt.plot(
    g_orbit_interp.r_pos * np.cos(g_orbit_interp.theta_pos),
    g_orbit_interp.r_pos * np.sin(g_orbit_interp.theta_pos),
    label="interp",
)
plt.plot(
    g_orbit_noninterp.r_pos * np.cos(g_orbit_noninterp.theta_pos),
    g_orbit_noninterp.r_pos * np.sin(g_orbit_noninterp.theta_pos),
    label="noninterp",
)
plt.legend()
plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel(r"r cos($\theta$)")
plt.ylabel(r"r sin($\theta$)")
plt.subplot(3, 3, 9)
plt.plot(g_orbit_interp.time, g_orbit_interp.magnetic_field_strength, label="interp")
plt.plot(
    g_orbit_noninterp.time, g_orbit_noninterp.magnetic_field_strength, label="noninterp"
)
plt.legend()
plt.xlabel(r"$t$")
plt.ylabel(r"$|B|$")
plt.tight_layout()
plt.show()
