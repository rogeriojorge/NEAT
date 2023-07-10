#!/usr/bin/env python3

import glob
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from simsopt.mhd import Vmec
from sympy import I

from neat.fields import StellnaQS
from neat.fields import Vmec as Vmec_NEAT
from neat.tracing import ChargedParticle, ParticleOrbit

"""                                                                           
Trace the orbit of a single particle in a
quasisymmetric stellarator using Near Axis and VMEC                 
"""

# Initialize an alpha particle

s_initial = 0.6  # psi/psi_a - VMEC radial coordinate
theta_initial = 0  # np.round(3*np.pi/2,2)   # initial poloidal angle - NEAT poloidal coordinate - theta_boozer - N phi_boozer
phi_initial = 0  # np.round(np.pi/2,2)       # initial toroidal angle - NEAT toroidal coordinate - phi_cyl on axis

energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
Lambda = 0.85  # = mu * B0 / energy
vpp_sign = -1  # initial sign of the parallel velocity, +1 or -1

# Simulation
nsamples = 100  # resolution in time
tfinal = 1e-8  # seconds

# Magnetic field
nfp = 4  # Number of field periods
B0 = 5.3267  # Tesla, magnetic field on-axis
constant_b20 = False  # Use a constant B20 (mean value) or the real function
Rmajor_ARIES = 7.7495  # Major radius
Rminor_ARIES = 1.7044  # Minor radius
s_boundary = 0.5  # Fraction of minor radius used for VMEC
r_avg = Rminor_ARIES * s_boundary  # Minor radius in VMEC

filename = f"input.nearaxis_sboundary{s_boundary}"
wout_filename = f"wout_nearaxis_sboundary{s_boundary}_000_000000.nc"

g_field = StellnaQS.from_paper(1, B0=B0, nphi=201)
# g_field = StellnaQS(rc=g_field_basis.rc*Rmajor_ARIES, zs=g_field_basis.zs*Rmajor_ARIES, etabar=g_field_basis.etabar/Rmajor_ARIES, B2c=g_field_basis.B2c*(B0/Rmajor_ARIES/Rmajor_ARIES),\
#                             B0=B0, nfp=g_field_basis.nfp, order='r3', nphi=111)

OUT_DIR = os.path.join(
    Path(__file__).parent.resolve(), f"comparison_qsc_vmec_orbits_s_b={s_boundary}"
)
os.makedirs(OUT_DIR, exist_ok=True)
minor_radius_plot = r_avg * np.sqrt(s_boundary)
filename_folder = os.path.join(OUT_DIR, filename)
g_field.to_vmec(
    filename=filename_folder,
    r=minor_radius_plot,
    params={"ntor": 7, "mpol": 7, "niter_array": [2000, 2000, 20000]},
    ntheta=14,
    ntorMax=7,
)  # standard ntheta=20, ntorMax=14
# if mayavi_loaded:
vmec = Vmec(filename_folder, verbose=True)
print("  Running VMEC")
vmec.run()

# subprocess.run([f"{os.path.join(os.path.dirname(__file__))}./xvmec2000", filename])
g_field_vmec = Vmec_NEAT(wout_filename=vmec.output_file, maximum_s=0.99)

g_particle = ChargedParticle(
    r_initial=r_avg * np.sqrt(s_initial),
    theta_initial=theta_initial,
    phi_initial=phi_initial,
    energy=energy,
    Lambda=Lambda,
    charge=charge,
    mass=mass,
    vpp_sign=vpp_sign,
)

phi_VMEC = g_field.to_RZ([[r_avg * np.sqrt(s_initial), theta_initial, phi_initial]])[2][
    0
]

g_particle_vmec = ChargedParticle(
    r_initial=s_initial,
    theta_initial=np.pi - theta_initial,
    phi_initial=phi_VMEC,
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
    g_particle_vmec,
    g_field_vmec,
    nsamples=nsamples,
    tfinal=tfinal,
    constant_b20=constant_b20,
)
total_time_vmec = time.time() - start_time_vmec
print(f"Finished in {total_time_vmec}s")

print("Creating parameter plot - NA")
g_orbit.plot(
    show=False,
    savefig="comparison_qsc_vmec_orbits_s_b="
    + str(s_boundary)
    + "/s_b="
    + str(s_boundary)
    + "_lambda="
    + str(Lambda)
    + "_s_i="
    + str(s_initial)
    + "_theta="
    + str(theta_initial)
    + "_phi="
    + str(phi_initial)
    + "_neat_parameters.png",
)

print("Creating 3D plot - NA")
g_orbit.plot_orbit_3d(
    show=False,
    r_surface=r_avg * np.sqrt(s_boundary),
    savefig="comparison_qsc_vmec_orbits_s_b="
    + str(s_boundary)
    + "/s_b="
    + str(s_boundary)
    + "_lambda="
    + str(Lambda)
    + "_s_i="
    + str(s_initial)
    + "_theta="
    + str(theta_initial)
    + "_phi="
    + str(phi_initial)
    + "_3d_neat.pdf",
)

print("Creating B contour plot - NA")
g_orbit.plot_orbit_contourB(show=False)

print("Creating parameter plot - VMEC")
g_orbit_vmec.plot(
    show=False,
    savefig="comparison_qsc_vmec_orbits_s_b="
    + str(s_boundary)
    + "/s_b="
    + str(s_boundary)
    + "_lambda="
    + str(Lambda)
    + "_s_i="
    + str(s_initial)
    + "_theta="
    + str(theta_initial)
    + "_phi="
    + str(phi_initial)
    + "_vmec_parameters.png",
)

print("Creating 3D plot - VMEC")
g_orbit_vmec.plot_orbit_3d(
    show=False,
    savefig="comparison_qsc_vmec_orbits_s_b="
    + str(s_boundary)
    + "/s_b="
    + str(s_boundary)
    + "_lambda="
    + str(Lambda)
    + "_s_i="
    + str(s_initial)
    + "_theta="
    + str(theta_initial)
    + "_phi="
    + str(phi_initial)
    + "_3d_vmec.pdf",
)

print("Creating B contour plot - VMEC")
g_orbit_vmec.plot_orbit_contourB(show=False)

# Animations
# print("Creating animation plot")
# g_orbit.plot_animation(show=True)

# print("Creating parameter plot 2")
# g_orbit_vmec.plot(show=False)

# print("Creating 3D plot 2")
# g_orbit_vmec.plot_orbit_3d(show=False)

# print("Creating animation plot 2")
# g_orbit_vmec.plot_animation(show=True)

print("Calculating differences between near axis and vmec - Cyl")
g_orbit.plot_diff_cyl(
    g_orbit_vmec,
    show=False,
    savefig="comparison_qsc_vmec_orbits_s_b="
    + str(s_boundary)
    + "/s_b="
    + str(s_boundary)
    + "_lambda="
    + str(Lambda)
    + "_s_i="
    + str(s_initial)
    + "_theta="
    + str(theta_initial)
    + "_phi="
    + str(phi_initial)
    + "_cyl.png",
)

print("Calculating differences between near axis and vmec - Boozer")
g_orbit.plot_diff_boozer(
    g_orbit_vmec,
    r_minor=(r_avg * np.sqrt(s_boundary)),
    MHD_code="VMEC",
    show=False,
    savefig="comparison_qsc_vmec_orbits_s_b="
    + str(s_boundary)
    + "/s_b="
    + str(s_boundary)
    + "_lambda="
    + str(Lambda)
    + "_s_i="
    + str(s_initial)
    + "_theta="
    + str(theta_initial)
    + "_phi="
    + str(phi_initial)
    + "_boozer.png",
)

print(s_boundary, Lambda, s_initial, theta_initial, phi_initial)

for objective_file in glob.glob(os.path.join(OUT_DIR, f"input.*")):
    os.remove(objective_file)
for objective_file in glob.glob(os.path.join(OUT_DIR, f"wout_*")):
    os.remove(objective_file)
for objective_file in glob.glob(os.path.join(OUT_DIR, f"threed1*")):
    os.remove(objective_file)
for objective_file in glob.glob(os.path.join(OUT_DIR, f"parvmecinfo*")):
    os.remove(objective_file)
