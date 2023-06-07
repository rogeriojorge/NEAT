#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from scipy.interpolate import CubicSpline as spline

from neat.tracing import ChargedParticleEnsemble, ParticleEnsembleOrbit_Vmec, ParticleEnsembleOrbit
from simsopt.mhd import Vmec
from neat.fields import StellnaQS, Vmec as Vmec_NEAT # isort:skip

import os  # isort:skip
import time  # isort:skip

import random

def rand_dist(start_in, end_in, num_in, seed):
    if seed==0: return np.linspace(start_in,end_in,num_in)
    else:    
        random.seed(seed)
        return np.array([random.uniform(start_in, end_in) for _ in range(num_in)])

def deviation_from_phis_vmec(phis_try):
    dev=np.zeros((ntheta,nphi))
    phis_try=np.reshape(phis_try,(ntheta, nphi))
    for i in np.arange(0, ntheta):
        for j in np.arange(0, nphi):
            phi_vmec=g_field.to_RZ([[r_avg*np.sqrt(s_initial),thetas_VMEC[i],phis_try[i][j]]])[2][0]
            dev[i][j]=np.mod(phi_vmec-phis_vmec[j],2*np.pi)
            dev[i][j]=np.min([dev[i][j],2*np.pi-dev[i][j]])
    return np.sum(dev)

"""                                                                           
Calculate the loss fraction of a distribution of particles
in a VMEC equilibrium                 
"""

# Initialize an ensemble of alpha particles at a radius = r_initial
# Calculate loss fraction at a radius = r_max
# Test OpenMP parallelization with an array of threads = nthreads_array
# The total number of particles is ntheta * nphi * (nlambda_passing+nlambda_trapped) * 2 (particles with v_parallel = +1, -1)
s_initial=0.5
s_max = 0.989
B0 = 5.3267         # Tesla, magnetic field on-axis (ARIES-CS)
energy = 3.52e6     # electron-volt
charge = 2          # times charge of proton
mass = 4            # times mass of proton     
nsamples = 50000    # resolution in time
tfinal = 1e-2       # seconds
constant_b20 =False # use a constant B20 (mean value) or the real function
s_boundary=1

ntheta = 5  # resolution in theta
nphi = 5  # resolution in phi
nlambda_trapped = 10  # number of pitch angles for trapped particles
nlambda_passing = 0  # number of pitch angles for passing particles
nthreads=4
dist=0 # 0 for linear distribution and other ints for random distributions

Rmajor_ARIES = 7.7495*3
Rminor_ARIES = 1.7044
r_avg=Rminor_ARIES*s_boundary

filename_vmec = f"input.nearaxis_sboundary{Rmajor_ARIES/r_avg}"
wout_filename_vmec = f"wout_nearaxis_sboundary{Rmajor_ARIES/r_avg}_000_000000.nc"
# wout_filename_vmec = f"wout_W7-X_standard_configuration.nc"

g_field_basis = StellnaQS.from_paper("r2 section 5.4", B0=B0, nphi=401)
g_field = StellnaQS(rc=g_field_basis.rc*Rmajor_ARIES, zs=g_field_basis.zs*Rmajor_ARIES, \
                    etabar=g_field_basis.etabar/Rmajor_ARIES, B2c=g_field_basis.B2c*(B0/Rmajor_ARIES/Rmajor_ARIES),\
                        B0=B0, nfp=g_field_basis.nfp, order='r3', nphi=401)

# g_field.to_vmec(filename=filename_vmec,r=r_avg, params={"ntor":8, "mpol":8, "niter_array":[10000,10000,20000],'ftol_array':[1e-13,1e-15,1e-17],'ns_array':[16,49,101]}, ntheta=48, ntorMax=48) #standard ntheta=20, ntorMax=14
# vmec=Vmec(filename=filename_vmec, verbose=True)
# vmec.run()

g_field_vmec = Vmec_NEAT(wout_filename=wout_filename_vmec,maximum_s=1.0)

g_particles = ChargedParticleEnsemble(
    r_initial=s_initial,
    r_max=s_max,
    energy=energy,
    charge=charge,
    mass=mass,
    ntheta=ntheta,
    nphi=nphi,
    nlambda_trapped=nlambda_trapped,
    nlambda_passing=nlambda_passing
)

thetas_VMEC=rand_dist(0,2*np.pi,ntheta,dist)
phis_vmec=rand_dist(np.pi/4,2*np.pi/g_field.nfp,nphi,dist)

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

# print(f"  Running with {2*(nlambda_passing+nlambda_trapped)*nphi*ntheta} particles took {total_time}s")
print(f"  Running with {g_orbits_vmec.nparticles} particles took {total_time}s")
print(f"  Final loss fraction = {g_orbits_vmec.total_particles_lost}")

phis_0 = np.full((ntheta, nphi), phis_vmec)
# print(phis_vmec)

print("Starting phis finder")
start_time = time.time()
phis_0_result=optimize.minimize(deviation_from_phis_vmec,phis_0, method='L-BFGS-B')#, tol=1e0)#options={'ftol':1e-8})
phis_0_x=phis_0_result.x
phis_0_x=np.reshape(phis_0_x,(ntheta, nphi))

# for i in np.arange(0, ntheta):
#     for j in np.arange(0, nphi):
#         phi_vmec=g_field.to_RZ([[r_avg*np.sqrt(s_initial),thetas_VMEC[i],phis_0_x[i][j]]])[2][0]
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
#         phis_vmec_new[i][j]=g_field.to_RZ([[r_avg*np.sqrt(s_initial),thetas_VMEC[i],phis_0_new[i][j]]])[2][0]
# print(phis_vmec_new)
# print(phis_vmec-phis_vmec_new)

thetas_VMEC_new = np.full((ntheta, nphi), thetas_VMEC.reshape((ntheta, 1)))
thetas_NA=np.pi-thetas_VMEC_new-(g_field.iota-g_field.iotaN)*varphis
# print(thetas_NA)
# thetas_NA=np.pi-thetas_VMEC
# print(thetas_NA)
g_particles.r_initial=r_avg*np.sqrt(s_initial)
g_particles.r_max=r_avg*np.sqrt(s_max)


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
    phis=varphis
)
total_time = time.time() - start_time
print(f"  Running with {g_orbits.nparticles} particles took {total_time}s")
print(f"  Final loss fraction = {g_orbits.total_particles_lost}")

g_orbits.loss_fraction(r_max=r_avg*np.sqrt(s_max), jacobian_weight=True)
g_orbits_vmec.loss_fraction(r_max=s_max, jacobian_weight=True)

plt.semilogx(g_orbits.time, g_orbits.loss_fraction_array,
    label="With jacobian weights and B20 constant - NA")
# plt.legend()
# plt.show()
plt.semilogx(g_orbits_vmec.time, g_orbits_vmec.loss_fraction_array,
    label="With jacobian weights and B20 constant - VMEC")
plt.legend()
plt.show()

for i in np.arange(0,4,1):
    plt.plot(g_orbits.time,(g_orbits.r_pos[i]/r_avg)**2,label=str(i)+" - NA", linestyle='dashdot', linewidth=3)
    # print(g_orbits.r_pos[i])
for i in np.arange(0,4,1):
    plt.plot(g_orbits_vmec.time,g_orbits_vmec.r_pos[i],label=str(i)+" - VMEC", linestyle='dotted', linewidth=3)
plt.legend()
plt.show()
# # plt.legend()
# # plt.show()

for i in np.arange(4,16,1):
    plt.plot(g_orbits.time,(g_orbits.r_pos[i]/r_avg)**2,label=str(i)+" - NA", linestyle='dashdot', linewidth=3)
for i in np.arange(4,16,1):
    plt.plot(g_orbits_vmec.time,g_orbits_vmec.r_pos[i],label=str(i)+" - VMEC", linestyle='dotted', linewidth=3)
plt.legend()
plt.show()
# # plt.legend()
# # plt.show()

for i in np.arange(16,24,1):
    plt.plot(g_orbits.time,(g_orbits.r_pos[i]/r_avg)**2,label=str(i)+" - NA", linestyle='dashdot', linewidth=3)
for i in np.arange(16,24,1):
    plt.plot(g_orbits_vmec.time,g_orbits_vmec.r_pos[i],label=str(i)+" - VMEC", linestyle='dotted', linewidth=3)
    # print(g_orbits_vmec.r_pos[i])
plt.legend()
plt.show()

for i in np.arange(0,g_orbits_vmec.nparticles,1):
    plt.plot(g_orbits.time,(g_orbits.r_pos[i]/r_avg)**2,label=str(i)+" - NA", linestyle='dashdot', linewidth=3)
for i in np.arange(0,g_orbits_vmec.nparticles,1):
    plt.plot(g_orbits_vmec.time,g_orbits_vmec.r_pos[i],label=str(i)+" - VMEC", linestyle='dotted', linewidth=3)
    # print(g_orbits_vmec.r_pos[i])
plt.legend()
plt.show()






