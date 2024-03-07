#!/usr/bin/env python

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from booz_xform import Booz_xform
from scipy.interpolate import CubicSpline as spline
from scipy.interpolate import interp1d
from simsopt.mhd import Vmec

from neat.fields import Boozxform, StellnaQS
from neat.fields import Vmec as VMEC_NEAT
from neat.tracing import ChargedParticle, ParticleOrbit

"""                                                                           
Trace the orbit of a single particle in a
vmec equilibrium                
"""

# Initialize an alpha particle at a radius = r_initial
r_initial = 0.3  # initial normalized toroidal magnetic flux (radial VMEC coordinate)
theta_initial = 0  # ;np.pi / 2  # initial poloidal angle
phi_initial = 0  # initial poloidal angle
energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
Lambda = 0.51  # = mu * B0 / energy
vpp_sign = 1  # initial sign of the parallel velocity, +1 or -1
nsamples = 3000  # resolution in time
tfinal = 1e-4  # seconds

B0 = 5.3267
# Scaling values
Rmajor_ARIES = 7.7495 * 2
Rminor_ARIES = 1.7044
Aspect=np.round(Rmajor_ARIES/Rminor_ARIES,2)

# filename = f"LandremanPaul2021_QA_scaled.nc"
# filename_vmec = f"input.nearaxis_sboundary{Rmajor_ARIES/Rminor_ARIES}_TRY"
# wout_filename = "wout_" + filename
# boozmn_filename = "boozmn_new_" + filename

# Names of input from NA and of VMEC and DESC outputs
filename_vmec = f"input.nearaxis_{Aspect}"
# wout_filename_desc = f"wout_nearaxis_{Rmajor_ARIES/Rminor_ARIES}_desc.nc"
wout_filename_vmec = f"wout_nearaxis_{Aspect}_000_000000.nc"

stellarator = ["precise QA", "2022 QH nfp4 well"]
g_field_basis = StellnaQS.from_paper(stellarator[0], B0=B0, nphi=401)
g_field_qsc = StellnaQS(
    rc=g_field_basis.rc * Rmajor_ARIES,
    zs=g_field_basis.zs * Rmajor_ARIES,
    etabar=g_field_basis.etabar / Rmajor_ARIES,
    B2c=g_field_basis.B2c * (B0 / Rmajor_ARIES / Rmajor_ARIES),
    B0=B0,
    nfp=g_field_basis.nfp,
    order="r3",
    nphi=401,
)

nu_array = g_field_qsc.varphi - g_field_qsc.phi
nu_spline_of_varphi = spline(
    np.append(g_field_qsc.varphi, 2 * np.pi / g_field_qsc.nfp),
    np.append(nu_array, nu_array[0]),
    bc_type="periodic",
)

phi0 = phi_initial - nu_spline_of_varphi(phi_initial)
phi_VMEC = g_field_qsc.to_RZ([[Rminor_ARIES * np.sqrt(r_initial), theta_initial, phi0]])[2][0]


# Creating wout of VMEC
g_field_qsc.to_vmec(filename=filename_vmec, r=Rminor_ARIES, params={"ntor":8, "mpol":8, \
    "niter_array":[10000,10000,20000],'ftol_array':[1e-13,1e-15,1e-16],'ns_array':[16,49,101]},
        ntheta=20, ntorMax=14) #standard ntheta=20, ntorMax=14
vmec=Vmec(filename=filename_vmec, verbose=True)
vmec.run()

g_field = VMEC_NEAT(wout_filename=wout_filename_vmec, maximum_s=1)
# b = Booz_xform()

# b.read_wout(wout_filename)
# # b.comput_surfs=100

# b.mboz = 100
# b.nboz = 100
# b.run()
# b.write_boozmn(boozmn_filename)

# g_field_booz = Boozxform(wout_filename=boozmn_filename)

# from scipy.io import netcdf_file
# net_file = netcdf_file(boozmn_filename, "r", mmap=False)
# bmnc = net_file.variables["iota_b"][()]
# # print(bmnc[1:])
# plt.plot(bmnc[1:])
# plt.show()
# print(bmnc[-1][0])
# print(np.average(bmnc[:][0]))
# plt.plot(np)

g_particle_qsc = ChargedParticle(
    r_initial=Rminor_ARIES * np.sqrt(r_initial),
    theta_initial=theta_initial,
    phi_initial=phi_initial,
    energy=energy,
    Lambda=Lambda,
    charge=charge,
    mass=mass,
    vpp_sign=-vpp_sign,
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

# g_particle_booz = ChargedParticle(
#     r_initial=r_initial,
#     theta_initial=theta_initial,
#     phi_initial=phi_initial,
#     energy=energy,
#     Lambda=Lambda,
#     charge=charge,
#     mass=mass,
#     vpp_sign=vpp_sign,
# )

print("Starting particle tracer")
start_time = time.time()
g_orbit_qsc = ParticleOrbit(
    g_particle_qsc, g_field_qsc, nsamples=nsamples, tfinal=tfinal, constant_b20=False
)
total_time = time.time() - start_time
print(f"Finished in {total_time}s")

print("Starting particle tracer")
start_time = time.time()
g_orbit = ParticleOrbit(g_particle, g_field, nsamples=nsamples, tfinal=tfinal)
total_time = time.time() - start_time
print(f"Finished in {total_time}s")

# print("Starting particle tracer")
# start_time = time.time()
# g_orbit_booz = ParticleOrbit(
#     g_particle_booz, g_field_booz, nsamples=nsamples, tfinal=tfinal
# )
# total_time = time.time() - start_time
# print(f"Finished in {total_time}s")

# print(len(g_field_qsc.to_RZ([Rminor_ARIES*np.sqrt(g_orbit_booz.r_pos),
#                     g_orbit_booz.theta_pos,
#                     g_orbit_booz.varphi_pos])[2])
# )
# norm_r_pos = (g_orbit_qsc.r_pos / (Rminor_ARIES)) ** 2
# plt.plot(norm_r_pos, label="qsc")
# plt.plot(g_orbit.r_pos, label="vmec")
# plt.plot(g_orbit_booz.r_pos, label="booz")
# plt.legend()
# # plt.show()

g_orbit.plot_orbit_contourB(show=False)
# g_orbit_booz.plot_orbit_contourB(show=True)

# g_orbit_qsc.plot(show=False)

# # # # print("Creating parameter plot")
# g_orbit.plot(show=False)

# # # # print("Creating parameter plot")
# g_orbit_booz.plot(show=False)

# # # g_orbit_booz.plot_diff_boozer(g_orbit,r_minor=Rminor_ARIES,show=True)

# g_orbit_booz.plot_diff_cyl(g_orbit, show=True)

# # # print("Creating 2D plot")
# g_orbit.plot_orbit(show=False)

# # print("Creating 2D plot")
# # g_orbit_booz.plot_orbit(show=False)

# # plt.show()

# # print("Creating 3D plot")
# # g_orbit_booz.plot_orbit_3d(show=False)

# print("Creating 3D plot")
# g_orbit.plot_orbit_3d(show=False)

# # print("Creating animation plot")
# # g_orbit.plot_animation(show=True)

# g_orbit.plot_orbit()
# g_orbit.plot()
# g_orbit.plot_diff_cyl(g_orbit_qsc)
# g_orbit_qsc.plot_diff_boozer(g_orbit, r_minor=Rminor_ARIES)
# np.savetxt("bench_modB.txt", g_orbit.magnetic_field_strength)
# np.savetxt("bench_cyl.txt", g_orbit.rpos_cylindrical)
# print("Creating animation plot")
# g_orbit.plot_animation(show=True)
