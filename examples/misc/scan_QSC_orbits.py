#!/usr/bin/env python3
import logging
import os
import subprocess
import time
from scipy import signal
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
from scipy.interpolate import interp1d, CubicSpline as spline
from sympy import I

from neat.fields import  StellnaQS
from neat.tracing import ChargedParticle, ParticleOrbit
from neat.plotting import butter_lowpass_filter, butter_lowpass_filter2

step_i=0.25
s_in=0.5
nfp=4
s_initials = np.round(np.arange(s_in,0.96,step_i), 2) # psi/psi_a for vmec
# s_initials = [0.1, 0.25,0.5, 0.75]

# Initial values for lambdas

lambdas=np.round(np.arange(0.96,0.999,0.01), 2) # = mu * B0 / energy

# Initial angular values

theta_initials=np.round(np.arange(0,2*np.pi,2*np.pi/2), 2)         # initial poloidal angle (vartheta=theta_Boozer - N phi_Boozer)
phi_initials=np.round(np.arange(0,2*np.pi/nfp,2*np.pi/nfp/8), 2)    # initial toroidal angle (cylindrical on axis)

# theta_initials=[0.25]#,1.1266087039264279,2.0149839496633657, 3.1415926535897931]
# phi_initials=[0.25]#,1.1266087039264279,2.0149839496633657, 3.1415926535897931]

# QA -> (theta, phi)= (0,0), (3.77,0)
# QH -> (theta, phi)= (0,0.94), (0.79,0.3), (1.26,0.93)

#Particle values
B0 = 5.3267         # Tesla, magnetic field on-axis (ARIES-CS)
energy = 3.52e6     # electron-volt
charge = 2          # times charge of proton
mass = 4            # times mass of proton
nsamples = 400   # resolution in time
tfinal = 1e-5       # seconds
constant_b20 = False # use a constant B20 (mean value) or the real function

# Scaling values
Rmajor_ARIES = 7.7495*2
Rminor_ARIES = 1.7044
r_avg=Rminor_ARIES

# Initializing and scaling NA field
stellarator = "2022 QH nfp4 well" 
g_field_basis = StellnaQS.from_paper(stellarator, B0=B0, nphi=101)
g_field = StellnaQS(rc=g_field_basis.rc*Rmajor_ARIES, zs=g_field_basis.zs*Rmajor_ARIES, \
                    etabar=g_field_basis.etabar/Rmajor_ARIES, B2c=g_field_basis.B2c*(B0/Rmajor_ARIES/Rmajor_ARIES),\
                        B0=B0, nfp=g_field_basis.nfp, order='r3', nphi=101)

print("Starting orbit scan")
start_time = time.time()

g_orbits = [
    ParticleOrbit(
        ChargedParticle(
            r_initial=r_avg*np.sqrt(s_initial),
            theta_initial=theta_initial-(g_field.iota-g_field.iotaN)*phi_initial,
            phi_initial=phi_initial,
            energy=energy,
            Lambda=lambdas[i],
            charge=charge,
            mass=mass,
            vpp_sign=vpp_sign,
        ),
        g_field,
        nsamples=nsamples,
        tfinal=tfinal,
        constant_b20=constant_b20
    )
    for i in np.arange(lambdas.size)
    for s_initial in s_initials
    for theta_initial in theta_initials
    for phi_initial in phi_initials
    for vpp_sign in [+1,-1]
]

total_time = time.time() - start_time
print(f"Finished in {total_time}s")

print('Creating plots.')
start_time = time.time()

folder=stellarator + '_QSC_A=' + str(Rmajor_ARIES/r_avg)
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams["figure.facecolor"] = "w"
plt.rcParams['lines.linewidth'] = 2

if not os.path.exists(folder): os.mkdir(folder)

norms_r_pos=[]
norms_r_pos_filt=[]

# plt.figure(figsize=(10, 6))

len_i=len(lambdas)
len_j=len(s_initials)
len_k=len(theta_initials)
len_l=len(phi_initials)
len_m=2


# for i in np.arange(0,len_i,1):  
#     # plt.figure(figsize=(10, 8))
#     lambda_str=str(lambdas[i])
#     for j in np.arange(0, len_j,1):
#         s_str=str(s_initials[j])
#         for k in np.arange(0, len_k,1):
#             theta_str=str(theta_initials[k])
#             for l in np.arange(0,len_l,1):
#                 varphi_str=str(phi_initials[l])
#                 for m in np.arange(0,len_m,1): 
#                     sign_str=('1' if m==0 else '-1')
    
#                     features_str='/' +lambda_str+ '_' + s_str+'_' +theta_str+'_' +varphi_str+'_' +sign_str+'_'
#                     g_orbits[i*len_i+j*len_j+k*len_k+m*len_m].plot_orbit_contourB(savefig=folder + features_str +'B_neat.png',show=False)
#                     # g_orbits[i*len_i+j*len_j+k*len_k+m*len_m].plot_diff_boozer(g_orbits[i*len_i+j*len_j+k*len_k+m*len_m],r_minor=r_avg,savefig=folder + features_str +'/diff_booz_neat_vmec')   print('NEAT-VMEC')
#                     # g_orbits[i*len_i+j*len_j+k*len_k+m*len_m].plot_diff_cyl(g_orbits[i*len_i+j*len_j+k*len_k+m*len_m],savefig=folder + features_str +'/diff_cyl_neat_vmec')
#                     g_orbits[i*len_i+j*len_j+k*len_k+m*len_m].plot( r_minor=r_avg, savefig=folder + features_str +'param_neat.png',show=False)
#                     g_orbits[i*len_i+j*len_j+k*len_k+m*len_m].plot_orbit_3d( r_surface=r_avg, savefig=folder + features_str +'orbit3d.png',show=False)
#                     plt.close()

for i, lambda_val in enumerate(lambdas):
    lambda_str = str(lambda_val)
    for j, s_initial in enumerate(s_initials):
        s_str = str(s_initial)
        for k, theta_initial in enumerate(theta_initials):
            theta_str = str(theta_initial)
            for l, phi_initial in enumerate(phi_initials):
                varphi_str = str(phi_initial)
                for m, vpp_sign in enumerate([+1, -1]):
                    sign_str = ('1' if m == 0 else '-1')
                    features_str = '/' + lambda_str + '_' + s_str + '_' + theta_str + '_' + varphi_str + '_' + sign_str + '_'
                    orbit_index = i * len(s_initials) * len(theta_initials) * len(phi_initials) * 2 + \
                                  j * len(theta_initials) * len(phi_initials) * 2 + \
                                  k * len(phi_initials) * 2 + \
                                  l * 2 + \
                                  m
                    g_orbits[orbit_index].plot_orbit_contourB(savefig=folder + features_str + 'B_neat.png', show=False)
                    g_orbits[orbit_index].plot(r_minor=r_avg, savefig=folder + features_str + 'param_neat.png', show=False)
                    g_orbits[orbit_index].plot_orbit_3d(r_surface=r_avg, savefig=folder + features_str + 'orbit3d.png', show=False)
                    plt.close()

total_time = time.time() - start_time
print(f"Finished in {total_time}s")
print('Plots saved in ' + folder + f' after {total_time}s"')