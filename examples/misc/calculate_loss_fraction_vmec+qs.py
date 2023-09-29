#!/usr/bin/env python

import random

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy.interpolate import CubicSpline as spline
from simsopt.mhd import Vmec

from neat.tracing import (
    ChargedParticleEnsemble,
    ParticleEnsembleOrbit,
    ParticleEnsembleOrbit_Vmec,
)

from neat.fields import StellnaQS, Vmec as Vmec_NEAT  # isort:skip

import os  # isort:skip
import time  # isort:skip


def rand_dist(start_in, end_in, num_in, seed):
    if seed == 0:
        return np.linspace(start_in, end_in, num_in)
    else:
        random.seed(seed)
        return np.array([random.uniform(start_in, end_in) for _ in range(num_in)])


def deviation_from_phis_vmec(phis_try):
    dev = np.zeros((ntheta, nphi))
    phis_try = np.reshape(phis_try, (ntheta, nphi))
    for i in np.arange(0, ntheta):
        for j in np.arange(0, nphi):
            phi_vmec = g_field.to_RZ(
                [[Rminor_ARIES * np.sqrt(s_initial), thetas_VMEC[i], phis_try[i][j]]]
            )[2][0]
            dev[i][j] = np.mod(phi_vmec - phis_vmec[j], 2 * np.pi)
            dev[i][j] = np.min([dev[i][j], 2 * np.pi - dev[i][j]])
    return np.sum(dev)

# def deviation_from_phis_vmec(phis_try):
#     phis_try = np.reshape(phis_try, (ntheta, nphi))
#     r=Rminor_ARIES * np.sqrt(s_initial)
#     phi_vmec_grid = np.vectorize(
#         lambda i, j: g_field.to_RZ(
#             [[r, thetas_VMEC[i], phis_try[i, j]]]
#         )[2][0]
#     )
#     i, j = np.meshgrid(np.arange(ntheta), np.arange(nphi), indexing="ij")
#     phi_vmec = phi_vmec_grid(i, j)
#     dev = np.mod(phi_vmec - phis_vmec[j], 2 * np.pi)
#     dev = np.minimum(dev, 2 * np.pi - dev)
#     return np.sum(dev)

# from multiprocessing import Pool
# import numpy as np

# def phi_vmec_ij(args):
#     i, r, thetas_VMEC, phis_try_ij = args
#     return g_field.to_RZ([[r, thetas_VMEC[i], phis_try_ij]])[2][0]

# def deviation_from_phis_vmec(phis_try):
#     phis_try = np.reshape(phis_try, (ntheta, nphi))
#     r = Rminor_ARIES * np.sqrt(s_initial)
    
#     i, j = np.meshgrid(np.arange(ntheta), np.arange(nphi), indexing="ij")
#     args = [(i_val, r, thetas_VMEC, phis_try[i_val, j_val]) for i_val, j_val in zip(i.flatten(), j.flatten())]
    
#     with Pool() as pool:
#         phi_vmec = np.array(pool.map(phi_vmec_ij, args)).reshape(ntheta, nphi)
    
#     dev = np.mod(phi_vmec - phis_vmec[j], 2 * np.pi)
#     dev = np.minimum(dev, 2 * np.pi - dev)
#     return np.sum(dev)

# from multiprocessing import Pool

# def phi_vmec_ij(args):
#     i, r, thetas_VMEC, phis_try_ij = args
#     return g_field.to_RZ([[r, thetas_VMEC[i], phis_try_ij]])[2][0]

# def deviation_from_phis_vmec(phis_try):
#     phis_try = np.reshape(phis_try, (ntheta, nphi))
#     r = Rminor_ARIES * np.sqrt(s_initial)
    
#     i, j = np.meshgrid(np.arange(ntheta), np.arange(nphi), indexing="ij")
#     args = [(i_val, r, thetas_VMEC, phis_try[i_val, j_val]) for i_val, j_val in zip(i.flatten(), j.flatten())]
    
#     with Pool() as pool:
#         phi_vmec = np.array(pool.map(phi_vmec_ij, args)).reshape(ntheta, nphi)
    
#     dev = np.mod(phi_vmec - phis_vmec[j], 2 * np.pi)
#     dev = np.minimum(dev, 2 * np.pi - dev)
#     return np.sum(dev)

"""                                                                           
Calculate the loss fraction of a distribution of particles
in a VMEC equilibrium                 
"""

# Initialize an ensemble of alpha particles at a radius = r_initial
# Calculate loss fraction at a radius = r_max
# Test OpenMP parallelization with an array of threads = nthreads_array
# The total number of particles is ntheta * nphi * (nlambda_passing+nlambda_trapped) * 2 (particles with v_parallel = +1, -1)


# Particles initial conditions
s_initial = 0.5
s_max = 0.989
energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton

# Integration settings
nsamples = 1000  # resolution in time
tfinal = 5e-4  # seconds

# Distribution
ntheta = 1  # resolution in theta
nphi = 15  # resolution in phi
nlambda_trapped = 3  # number of pitch angles for trapped particles
nlambda_passing = 0  # number of pitch angles for passing particles
nthreads = 4
dist = 0  # 0 for linear distribution and other ints for random distributions

# Field Scaling Factors
B0 = 5.3267  # Tesla, magnetic field on-axis (ARIES-CS)
Rmajor_ARIES = 7.7495 * 2
Rminor_ARIES = 1.7044
constant_b20 = False  # use a constant B20 (mean value) or the real function
Aspect_r=np.round(Rmajor_ARIES/Rminor_ARIES,2)

stellarator = ["precise QA", "2022 QH nfp4 well"]

# Names of input from NA and output from VMEC
filename_vmec = f"input.nearaxis_{Aspect_r}_QA"
wout_filename_vmec = f"wout_nearaxis_{Aspect_r}_QA_000_000000.nc"

# Initializing and scaling NA field
g_field_basis = StellnaQS.from_paper(stellarator[0], B0=B0, nphi=101)
g_field = StellnaQS(
    rc=g_field_basis.rc * Rmajor_ARIES,
    zs=g_field_basis.zs * Rmajor_ARIES,
    etabar=g_field_basis.etabar / Rmajor_ARIES,
    B2c=g_field_basis.B2c * (B0 / Rmajor_ARIES / Rmajor_ARIES),
    B0=B0,
    nfp=g_field_basis.nfp,
    order="r3",
    nphi=401,
)

# Creating wout of VMEC
# g_field.to_vmec(
#     filename=filename_vmec,
#     r=Rminor_ARIES,
#     params={
#         "ntor": 8,
#         "mpol": 8,
#         "niter_array": [10000, 10000, 20000],
#         "ftol_array": [1e-13, 1e-16, 1e-18],
#         "ns_array": [16, 49, 101],
#     },
#     ntheta=48,
#     ntorMax=48,
# )  # standard ntheta=20, ntorMax=14
# vmec = Vmec(filename=filename_vmec, verbose=True)
# vmec.run()

# Initializing field with wout from VMEC
g_field_vmec = Vmec_NEAT(wout_filename=wout_filename_vmec, maximum_s=1.0)

# Initializing ensemble of particles
g_particles = ChargedParticleEnsemble(
    r_initial=s_initial,
    r_max=s_max,
    energy=energy,
    charge=charge,
    mass=mass,
    ntheta=ntheta,
    nphi=nphi,
    nlambda_trapped=nlambda_trapped,
    nlambda_passing=nlambda_passing,
)

# Creatig a linspace for dist=0 and random with seed=distro for ditro!=0
thetas_VMEC = rand_dist(0, 2 * np.pi, ntheta, dist)
phis_vmec = rand_dist(0, 2 * np.pi / g_field.nfp, nphi, dist)

# Tracing orbits
print("Starting particle tracer - VMEC")
start_time = time.time()
g_orbits_vmec = ParticleEnsembleOrbit_Vmec(
    g_particles,
    g_field_vmec,
    tfinal=tfinal,
    nsamples=nsamples,
    nthreads=nthreads,
    dist=dist,
)
total_time = time.time() - start_time

print(f"  Running with {g_orbits_vmec.nparticles} particles took {total_time}s")
print(f"  Final loss fraction = {g_orbits_vmec.total_particles_lost}")

phis_0 = np.full((ntheta, nphi), phis_vmec)


print("Starting phis finder")
start_time = time.time()
phis_0_result = optimize.minimize(
    deviation_from_phis_vmec, phis_0, method="L-BFGS-B"
)  # , tol=1e0)#options={'ftol':1e-8})
phis_0_x = phis_0_result.x
phis_0_x = np.reshape(phis_0_x, (ntheta, nphi))

# To check results
# for i in np.arange(0, ntheta):
#     for j in np.arange(0, nphi):
#         phi_vmec=g_field.to_RZ([[Rminor_ARIES*np.sqrt(s_initial),thetas_VMEC[i],phis_0_x[i][j]]])[2][0]
#         if phi_vmec<0:
#             phi_vmec=2*np.pi+phi_vmec
#         b=np.abs(phis_vmec[j] - phi_vmec)
#         c=np.min([b,2*np.pi-b])
#         print(c)

total_time = time.time() - start_time
print(f"  Running transforms took {total_time}s")

nu_array = g_field.varphi - g_field.phi

nu_spline_of_phi = spline(
    np.append(g_field.phi, 2 * np.pi / g_field.nfp),
    np.append(nu_array, nu_array[0]),
    bc_type="periodic",
)

varphis = phis_0_x + nu_spline_of_phi(phis_0_x)

# To check results
# nu_array2 = g_field.varphi - g_field.phi
# nu_spline_of_varphi = spline(
#     np.append(g_field.varphi, 2 * np.pi / g_field.nfp),
#     np.append(nu_array2, nu_array2[0]),
#     bc_type="periodic",
# )

# phis_0_new = varphis - nu_spline_of_varphi(varphis)
# phis_vmec_new = np.zeros_like(phis_0_new)
# for i in np.arange(0, ntheta):
#     for j in np.arange(0, nphi):
#         phis_vmec_new[i][j]=g_field.to_RZ([[Rminor_ARIES*np.sqrt(s_initial),thetas_VMEC[i],phis_0_new[i][j]]])[2][0]
# print(phis_vmec_new)
# print(phis_vmec-phis_vmec_new)

thetas_VMEC_new = np.full((ntheta, nphi), thetas_VMEC.reshape((ntheta, 1)))
thetas_NA = np.pi - thetas_VMEC_new - (g_field.iota - g_field.iotaN) * varphis

# thetas_NA=np.pi-thetas_VMEC

g_particles.r_initial = Rminor_ARIES * np.sqrt(s_initial)
g_particles.r_max = Rminor_ARIES * np.sqrt(s_max)

# print(varphis, thetas)

print("Starting particle tracer - NA")
start_time = time.time()
g_orbits = ParticleEnsembleOrbit(
    g_particles,
    g_field,
    tfinal=tfinal,
    nsamples=nsamples,
    nthreads=nthreads,
    constant_b20=constant_b20,
    dist=dist,
    thetas=thetas_NA,
    phis=varphis,
)
total_time = time.time() - start_time
print(f"  Running with {g_orbits.nparticles} particles took {total_time}s")
print(f"  Final loss fraction = {g_orbits.total_particles_lost}")

g_orbits.loss_fraction(r_max=Rminor_ARIES * np.sqrt(s_max), jacobian_weight=False)
g_orbits_vmec.loss_fraction(r_max=s_max, jacobian_weight=False)

# plt.rc('xtick', labelsize=14)
# plt.rc('ytick', labelsize=14)
# plt.rc('font', size=14)
# plt.rc('legend', fontsize=14)	
plt.rc('lines', linewidth=5)
plt.figure(figsize=(20,10))
# fig,ax=plt.subplots(1,1,figsize=(20,10))
# ax.xaxis.offsetText.set_fontsize(50)
plt.tick_params(axis='x', labelsize=50,pad=20)
plt.tick_params(axis='y', labelsize=50,pad=10)
plt.semilogx(
    g_orbits.time,
    100*np.array(g_orbits.loss_fraction_array),
    label=" NA ",
)
plt.semilogx(
    g_orbits_vmec.time,
    100*np.array(g_orbits_vmec.loss_fraction_array),
    label=" VMEC ",
)
plt.legend(loc='best', fontsize=50)
plt.xlabel(r"$t \ (s)$",fontsize=60,labelpad=20)
plt.ylabel(r"Loss fraction (%)",fontsize=60,labelpad=20)
plt.tight_layout()
plt.savefig(f"results_losses/loss_{s_initial}.pdf")
# plt.show()
plt.close()

for i in np.arange(0, 4, 1):
    plt.plot(
        g_orbits.time,
        (g_orbits.r_pos[i] / Rminor_ARIES) ** 2,
        label=str(i) + " - NA",
        linestyle="dashdot",
        linewidth=3,
    )
    # print(g_orbits.r_pos[i])
for i in np.arange(0, 4, 1):
    plt.plot(
        g_orbits_vmec.time,
        g_orbits_vmec.r_pos[i],
        label=str(i) + " - VMEC",
        linestyle="dotted",
        linewidth=3,
    )
plt.legend()
# plt.show()
# # plt.legend()
# # plt.show()

for i in np.arange(4, 16, 1):
    plt.plot(
        g_orbits.time,
        (g_orbits.r_pos[i] / Rminor_ARIES) ** 2,
        label=str(i) + " - NA",
        linestyle="dashdot",
        linewidth=3,
    )
for i in np.arange(4, 16, 1):
    plt.plot(
        g_orbits_vmec.time,
        g_orbits_vmec.r_pos[i],
        label=str(i) + " - VMEC",
        linestyle="dotted",
        linewidth=3,
    )
plt.legend()
# plt.show()

for i in np.arange(16, 24, 1):
    plt.plot(
        g_orbits.time,
        (g_orbits.r_pos[i] / Rminor_ARIES) ** 2,
        label=str(i) + " - NA",
        linestyle="dashdot",
        linewidth=3,
    )
for i in np.arange(16, 24, 1):
    plt.plot(
        g_orbits_vmec.time,
        g_orbits_vmec.r_pos[i],
        label=str(i) + " - VMEC",
        linestyle="dotted",
        linewidth=3,
    )
plt.legend()
# plt.show()
plt.close()

plt.figure(figsize=(20,10))
fig,ax=plt.subplots(1,1,figsize=(20,10))
ax.xaxis.offsetText.set_fontsize(50)
plt.ticklabel_format(axis='x', style='sci', scilimits=(4,-4))
plt.tick_params(axis='x', labelsize=50,pad=10)
plt.tick_params(axis='y', labelsize=50,pad=10)
for i in np.arange(0, g_orbits_vmec.nparticles, 1):
    if i!=0:
        plt.plot(
            g_orbits.time,
            (g_orbits.r_pos[i] / Rminor_ARIES) ** 2,
            linestyle="dashdot",
            linewidth=3,
        )
# for i in np.arange(0, g_orbits_vmec.nparticles, 1):
        plt.plot(
            g_orbits_vmec.time,
            g_orbits_vmec.r_pos[i],
            linestyle="dotted",
        )
    else:
        plt.plot(
            g_orbits.time,
            (g_orbits.r_pos[i] / Rminor_ARIES) ** 2,
            "k-.",
            label=" NA ",
        )
        plt.plot(
            g_orbits_vmec.time,
            g_orbits_vmec.r_pos[i],
            "k:",
            label=" VMEC ",
        )
plt.legend(loc='best', fontsize=50)
plt.xlabel(r"$t \ (s)$",fontsize=60,labelpad=20)
plt.ylabel(r"s",fontsize=60,labelpad=20)
plt.tight_layout()
plt.savefig(f"results_losses/orbits_all_{s_initial}.pdf")
# plt.show()
