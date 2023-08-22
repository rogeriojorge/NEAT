#!/usr/bin/env python3

import glob
import logging
import os
import subprocess
import time
from pathlib import Path

import desc.io  # used for loading and saving data
import matplotlib.pyplot as plt
import numpy as np
from desc.equilibrium import Equilibrium
from desc.objectives import get_fixed_boundary_constraints
from desc.vmec import (
    VMECIO,
)  # used to save equilibrium objects as VMEC-formatted .nc files
from mpi4py import MPI
from simsopt.mhd import Vmec
from simsopt.util import MpiPartition
from sympy import I

from neat.fields import StellnaQS
from neat.fields import Vmec as Vmec_NEAT
from neat.tracing import ChargedParticle, ParticleOrbit

"""                                                                           
Trace the orbit of a single particle in a
quasisymmetric stellarator using Near Axis and VMEC                 
"""

# Initialize an alpha particle

s_initial = 0.25  # psi/psi_a - VMEC radial coordinate
theta_initial = 1.26  # np.round(3*np.pi/2,2)   # initial poloidal angle - NEAT poloidal coordinate - theta_boozer - N phi_boozer
phi_initial = 0.31  # np.round(np.pi/2,2)       # initial toroidal angle - NEAT toroidal coordinate - phi_cyl on axis

energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
Lambda = 0.1  # = mu * B0 / energy
vpp_sign = -1  # initial sign of the parallel velocity, +1 or -1

# Simulation
nsamples = 1000  # resolution in time
tfinal = 1e-5  # seconds

# Magnetic field
B0 = 5.3267  # Tesla, magnetic field on-axis
constant_b20 = False  # Use a constant B20 (mean value) or the real function
Rmajor_ARIES = 7.7495*2  # Major radius double double of ARIES-CS
Rminor_ARIES = 1.7044  # Minor radius
Aspect=np.round(Rmajor_ARIES/Rminor_ARIES,2)

filename = f"input.nearaxis_{Aspect}_desc"
wout_filename = f"wout_nearaxis_{Aspect}_desc.nc"

g_field_basis = StellnaQS.from_paper("precise QA", B0=B0, nphi=201)
g_field = StellnaQS(
    rc=g_field_basis.rc * Rmajor_ARIES,
    zs=g_field_basis.zs * Rmajor_ARIES,
    etabar=g_field_basis.etabar / Rmajor_ARIES,
    B2c=g_field_basis.B2c * (B0 / Rmajor_ARIES / Rmajor_ARIES),
    B0=B0,
    nfp=g_field_basis.nfp,
    order="r3",
    nphi=111,
)

# Folder creation for results
OUT_DIR = os.path.join(
    Path(__file__).parent.resolve(), f"comparison_qsc_desc_orbits_{Aspect}"
)
os.makedirs(OUT_DIR, exist_ok=True)
filename_folder = os.path.join(OUT_DIR, filename)

# Creating a DESC object from pyQsc near acis
g_field_desc = Equilibrium.from_near_axis(
    g_field,  # the Qsc equilibrium object
    r=Rminor_ARIES,  # the finite radius (m) at which to evaluate the Qsc surface to use as the DESC boundary
    L=8,  # DESC radial resolution
    M=8,  # DESC poloidal resolution
    N=8,  # DESC toroidal resolution
)


constraints = get_fixed_boundary_constraints(iota=False) # get the fixed-boundary constraints, ( pressure and current profile (iota=False flag means fix current)
# solve the equilibrium
g_field_desc.solve(verbose=3, ftol=1e-2,objective="force",maxiter=100,xtol=1e-6,constraints=constraints)
VMECIO.save(g_field_desc, wout_filename)

g_field_vmec = Vmec_NEAT(wout_filename=wout_filename, maximum_s=1)

g_particle = ChargedParticle(
    r_initial=Rminor_ARIES * np.sqrt(s_initial),
    theta_initial=theta_initial,
    phi_initial=phi_initial,
    energy=energy,
    Lambda=Lambda,
    charge=charge,
    mass=mass,
    vpp_sign=vpp_sign,
)

phi_initial_vmec = g_field.to_RZ(
    [[Rminor_ARIES * np.sqrt(s_initial), theta_initial, phi_initial]]
)[2][0]

g_particle_vmec = ChargedParticle(
    r_initial=s_initial,
    theta_initial=-theta_initial,
    phi_initial=phi_initial_vmec,
    energy=energy,
    Lambda=Lambda,
    charge=charge,
    mass=mass,
    vpp_sign=vpp_sign,
)

print("Starting particle tracer 1")
start_time = time.time()
g_orbit = ParticleOrbit(
    g_particle, 
    g_field,
    nsamples=nsamples, 
    tfinal=tfinal, 
    constant_b20=constant_b20,
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
    savefig=f"comparison_qsc_desc_orbits_{Aspect}"
    + "/aspect="
    + str(Aspect)
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

# print("Creating 2D plot")
# g_orbit.plot_orbit(show=False)

print("Creating 3D plot - NA")
g_orbit.plot_orbit_3d(
    show=False,
    r_surface=Rminor_ARIES,
    savefig=f"comparison_qsc_desc_orbits_{Aspect}"
    + "/aspect"
    + str(Aspect)
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
    savefig=f"comparison_qsc_desc_orbits_{Aspect}"
    + "/aspect"
    + str(Aspect)
    + "_lambda="
    + str(Lambda)
    + "_s_i="
    + str(s_initial)
    + "_theta="
    + str(theta_initial)
    + "_phi="
    + str(phi_initial)
    + "_desc_parameters.png",
)

# print("Creating 2D plot")
# g_orbit.plot_orbit(show=False)

# print("Creating 3D plot - VMEC")
g_orbit_vmec.plot_orbit_3d(
    show=False,
    savefig=f"comparison_qsc_desc_orbits_{Aspect}"
    + "/aspect"
    + str(Aspect)
    + "_lambda="
    + str(Lambda)
    + "_s_i="
    + str(s_initial)
    + "_theta="
    + str(theta_initial)
    + "_phi="
    + str(phi_initial)
    + "_3d_desc.pdf",
)

print("Creating B contour plot - VMEC")
g_orbit_vmec.plot_orbit_contourB(show=False)

# Animations
print("Creating animation plot")
g_orbit.plot_animation(show=True, r_surface=Rminor_ARIES * np.sqrt(s_initial))

print("Creating parameter plot 2")
g_orbit_vmec.plot(show=False)

print("Creating 3D plot 2")
g_orbit_vmec.plot_orbit_3d(show=False)

print("Creating animation plot 2")
g_orbit_vmec.plot_animation(show=True)

print("Calculating differences between near axis and vmec - Cyl")
g_orbit.plot_diff_cyl(
    g_orbit_vmec,
    show=False,
    savefig=f"comparison_qsc_desc_orbits_{Aspect}"
    + "/aspect"
    + str(Aspect)
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
    r_minor=(Rminor_ARIES),
    show=False,
    savefig=f"comparison_qsc_desc_orbits_{Aspect}"
    + "/aspect"
    + str(Aspect)
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

print(Lambda, s_initial, theta_initial, phi_initial)

# for objective_file in glob.glob(os.path.join(OUT_DIR,f"input.*")): os.remove(objective_file)
# for objective_file in glob.glob(os.path.join(OUT_DIR,f"wout_*")): os.remove(objective_file)
# for objective_file in glob.glob(os.path.join(OUT_DIR,f"threed1*")): os.remove(objective_file)
# for objective_file in glob.glob(os.path.join(OUT_DIR,f"parvmecinfo*")): os.remove(objective_file)
