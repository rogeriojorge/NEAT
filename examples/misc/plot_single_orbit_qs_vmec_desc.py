#!/usr/bin/env python3

import os
import time
from pathlib import Path

# import desc.io  # used for loading and saving data
import matplotlib.pyplot as plt
import numpy as np

from mpi4py import MPI
from simsopt.mhd import Vmec
from simsopt.util import MpiPartition
from sympy import I
from scipy.interpolate import interp1d, CubicSpline as spline

from desc.equilibrium import Equilibrium
from desc.vmec import VMECIO
from desc.objectives import get_fixed_boundary_constraints, get_NAE_constraints

from neat.fields import StellnaQS
from neat.fields import Vmec as Vmec_NEAT
from neat.fields import Boozxform
from neat.tracing import ChargedParticle, ParticleOrbit

"""                                                                           
Trace the orbit of a single particle in a
quasisymmetric stellarator using Near Axis and VMEC                 
"""

# Initialize an alpha particle

s_initial = 0.25  # psi/psi_a - VMEC radial coordinate
theta_initial = 0.1  # np.round(3*np.pi/2,2)   # initial poloidal angle - NEAT poloidal coordinate - theta_boozer - N phi_boozer
phi_initial = 0.1  # np.round(np.pi/2,2)       # initial toroidal angle - NEAT toroidal coordinate - phi_cyl on axis

energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
Lambda = 0.98  # = mu * B0 / energy
vpp_sign = -1  # initial sign of the parallel velocity, +1 or -1

# Simulation
nsamples = 10000  # resolution in time
tfinal = 1e-3  # seconds

# Magnetic field
B0 = 5.3267  # Tesla, magnetic field on-axis
constant_b20 = False  # Use a constant B20 (mean value) or the real function
Rmajor_ARIES = 7.7495*2  # Major radius double double of ARIES-CS
Rminor_ARIES = 1.7044  # Minor radius
Aspect=np.round(Rmajor_ARIES/Rminor_ARIES,2)

# Names of input from NA and of VMEC and DESC outputs
filename_vmec = f"input.nearaxis_{Aspect}_QA"
wout_filename_desc = f"wout_nearaxis_{Aspect}_QA_desc.nc"
wout_filename_vmec = f"wout_nearaxis_{Aspect}_QA_000_000000.nc"
boozmn_filename = f"boozmn_new_nearaxis_{Aspect}_QA_000_000000.nc" 

# List of working stellarators
stellarator = ["precise QA", "2022 QH nfp4 well"]

g_field_basis = StellnaQS.from_paper(stellarator[0], B0=B0, nphi=201)
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

nu_array = g_field.varphi - g_field.phi
nu_spline_of_varphi = spline(
    np.append(g_field.varphi, 2 * np.pi / g_field.nfp),
    np.append(nu_array, nu_array[0]),
    bc_type="periodic",
)

# Folder creation for results
OUT_DIR = os.path.join(
    Path(__file__).parent.resolve(), f"comparison_qsc_desc_orbits_{Aspect}"
)
os.makedirs(OUT_DIR, exist_ok=True)
filename_folder = os.path.join(OUT_DIR, filename_vmec)

# Creating a DESC object from pyQsc near acis
# # g_field_desc = Equilibrium.from_near_axis(
# #     g_field,  # the Qsc equilibrium object
# #     r=Rminor_ARIES,  # the finite radius (m) at which to evaluate the Qsc surface to use as the DESC boundary
# #     L=16,  # DESC radial resolution
# #     M=8,  # DESC poloidal resolution
# #     N=8,  # DESC toroidal resolution
# #     ntheta=75
# # )
# constraints = get_fixed_boundary_constraints(iota=False) # get the fixed-boundary constraints, ( pressure and current profile (iota=False flag means fix current)
# # solve the equilibrium
# g_field_desc.solve(verbose=3, ftol=1e-2,objective="force",maxiter=100,xtol=1e-6,constraints=constraints)
# VMECIO.save(g_field_desc, wout_filename_desc)

# Using a desc equilibria but in the VMEC format
g_field_vmec = Vmec_NEAT(wout_filename= wout_filename_vmec, maximum_s=1)
# g_field_desc = Vmec_NEAT(wout_filename= wout_filename_desc, maximum_s=1)
g_field_booz = Boozxform(wout_filename=boozmn_filename)


g_particle = ChargedParticle(
    r_initial=Rminor_ARIES * np.sqrt(s_initial),
    theta_initial=theta_initial-(g_field.iota-g_field.iotaN)*phi_initial,
    phi_initial=phi_initial,
    energy=energy,
    Lambda=Lambda,
    charge=charge,
    mass=mass,
    vpp_sign=vpp_sign,
)

phi0 = phi_initial - nu_spline_of_varphi(phi_initial)
phi_VMEC=g_field.to_RZ([[Rminor_ARIES*np.sqrt(s_initial),theta_initial,phi0]])[2][0]

g_particle_vmec = ChargedParticle(
    r_initial=s_initial,
    theta_initial = np.pi-(theta_initial),
    phi_initial = phi_VMEC,
    energy=energy,
    Lambda=Lambda,
    charge=charge,
    mass=mass,
    vpp_sign=vpp_sign,
)

g_particle_booz = ChargedParticle(
    r_initial=s_initial,
    theta_initial = theta_initial,     
    phi_initial = phi_initial,
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


print("Starting particle tracer 4")
start_time_booz = time.time()
g_orbit_booz = ParticleOrbit(
    g_particle_booz, 
    g_field_booz, 
    nsamples=nsamples, 
    tfinal=tfinal
)
total_time_booz = time.time() - start_time_booz
print(f"Finished in {total_time_booz}s")


plt.style.use('seaborn-v0_8-white')
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams["figure.facecolor"] = "w"
plt.rcParams['lines.linewidth'] = 3
plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24)
plt.rc('font', size=24)
plt.rc('legend', fontsize=18)
plt.rc('lines', linewidth=3)

print("Creating parameter plot - NA")
g_orbit.plot(
    r_minor=Rminor_ARIES,
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
    + "_NA_parameters.png",
)

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
    + "_3d_NA.png",
)

print("Creating B contour plot - NA")
g_orbit.plot_orbit_contourB(
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
    + "_NA_Contour.png",
    )

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
    + "_vmec_parameters.png",
)

print("Creating 3D plot - VMEC")
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
    + "_3d_vmec.png",
)

print("Creating B contour plot - VMEC")
g_orbit_vmec.plot_orbit_contourB(
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
    + "_VMEC_Contour.png",
    )

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
    + "_vmec_parameters.png",
)

print("Creating 3D plot - VMEC")
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
    + "_3d_vmec.png",
)

print("Creating B contour plot - VMEC")
g_orbit_vmec.plot_orbit_contourB(
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
    + "_VMEC_Contour.png",
    )


# Animations
# print("Creating animation plot - NA")
# g_orbit.plot_animation(show=False, r_surface=Rminor_ARIES * np.sqrt(s_initial))

# print("Creating animation plot - booz")
# g_orbit_booz.plot_animation(show=False)

print("Calculating differences between near axis and booz - Cyl")
g_orbit.plot_diff_cyl(
    g_orbit_booz,
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

print("Calculating differences between NA and booz - Boozer")
g_orbit.plot_diff_boozer(
    g_orbit_booz,
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
