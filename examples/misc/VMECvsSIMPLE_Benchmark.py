#!/usr/bin/env python
import time
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

from neat.fields import Vmec as VMEC_NEAT, Simple
from neat.tracing import ChargedParticle, ParticleOrbit

"""                                                                           
Perform a benchmark on the particle tracing with Gyronimo vs SIMPLE              
"""

# Initialize an alpha particle at a radius = r_initial
r_initial = 0.25  # initial normalized toroidal magnetic flux (radial VMEC coordinate)
theta_initial = 0.1  # initial poloidal angle
phi_initial = 0.1  # initial poloidal angle
energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
Lambda = [0.8,0.95,0.99]  # = mu * B0 / energy
vpp_sign = -1  # initial sign of the parallel velocity, +1 or -1
nsamples = [50000,50001,50002]  # resolution in time
tfinal = [1e-2,1e-2,1e-2]  # seconds
linewidth = [1.5,1.5,1.5]

B0 = 5.3267
Rmajor_ARIES = 7.7494
Rminor_ARIES = 1.7044
Aspect_ratios=Rmajor_ARIES/ Rminor_ARIES
iterator=range(nsamples.size)

filename = "Matt_precise_wout"
wout_filename = "NEAT/examples/misc/wout_Matt_nfp2_QA_rescaled.nc"

g_field_vmec = VMEC_NEAT(wout_filename=wout_filename)
g_field_simple = Simple(wout_filename=wout_filename, ns_s=5, ns_tp=5, multharm=3)

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

time_vmec = []
time_simple = []
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('font', size=14)
plt.rc('legend', fontsize=14)	
for j in iterator:
    g_particle = ChargedParticle(
    	r_initial=r_initial,
    	theta_initial=theta_initial,
    	phi_initial=phi_initial,
    	energy=energy,
    	Lambda=Lambda[j],
    	charge=charge,
    	mass=mass,
    	vpp_sign=vpp_sign,
    )
 
    print("=" * 80)
    print(f"nsamples = {nsamples[j]}")
    
    print("  Starting particle tracer vmec")
    start_time = time.time()
    g_particle.theta_initial = theta_initial
    g_particle.phi_initial = phi_initial
    g_orbit_vmec = ParticleOrbit(
        g_particle, g_field_vmec, nsamples=nsamples[-1], tfinal=tfinal[-1]
    )
    total_time = time.time() - start_time
    print(f"  Finished vmec in {total_time}s")
    time_vmec.append(total_time)
    
    print("  Starting particle tracer simple")
    start_time = time.time()
    g_particle.theta_initial = - theta_initial
    g_particle.phi_initial = - phi_initial
    g_orbit_simple = ParticleOrbit(
        g_particle, g_field_simple, nsamples=nsamples[-1], tfinal=tfinal[-1]
    )
    total_time = time.time() - start_time
    print(f"  Finished simple in {total_time}s")
    time_simple.append(total_time)

    ##############################################################################################

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 3, 1)
    plt.plot(g_orbit_vmec.time, np.pi - g_orbit_vmec.theta_pos, label="vmec")
    plt.plot(g_orbit_simple.time, np.pi + g_orbit_simple.theta_pos, label="simple")
    plt.legend()
    plt.xlabel(r"$t (s)$")
    plt.ylabel(r"$\theta$")
    plt.subplot(2, 3, 2)
    plt.plot(g_orbit_vmec.time, g_orbit_vmec.varphi_pos, label="vmec")
    plt.plot(g_orbit_simple.time, -g_orbit_simple.varphi_pos, label="simple")
    plt.legend()
    plt.xlabel(r"$t (s)$")
    plt.ylabel(r"$\varphi$")
    plt.subplot(2, 3, 3)
    plt.plot(g_orbit_vmec.time, g_orbit_vmec.v_parallel, label="vmec")
    plt.plot(g_orbit_simple.time, -g_orbit_simple.v_parallel, label="simple")
    plt.legend()
    plt.xlabel(r"$t (s)$")
    plt.ylabel(r"$v_\parallel$")
    plt.subplot(2, 3, 4)
    plt.plot(
        g_orbit_vmec.time,
        (g_orbit_vmec.total_energy - g_orbit_vmec.total_energy[0])
        / g_orbit_vmec.total_energy[0],
        label="vmec",
    )
    plt.legend()
    plt.xlabel(r"$t (s)$")
    plt.ylabel(r"$(E-E_0)/E_0$")
    plt.subplot(2, 3, 5)
    plt.plot(g_orbit_vmec.time, g_orbit_vmec.rdot, "r-", label=r"$\dot r$ vmec")
    plt.plot(g_orbit_vmec.time, -g_orbit_vmec.thetadot, "g-", label=r"$\dot \theta$ vmec")
    plt.plot(g_orbit_vmec.time, g_orbit_vmec.varphidot, "m-", label=r"$\dot \varphi$ vmec")
    plt.xlabel(r"$t (s)$")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.subplots_adjust(right=0.8)  # Adjust the right margin to fit the legend
    plt.subplot(2, 3, 6)
    plt.plot(g_orbit_vmec.time, g_orbit_vmec.magnetic_field_strength, label="vmec")
    plt.plot(g_orbit_simple.time, g_orbit_simple.magnetic_field_strength, label="simple") #We do not get this value
    plt.legend()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$|B|$")
    plt.tight_layout()
    # plt.show()
    plt.savefig('NEAT/examples/misc/results/Stuff_' + filename + str(nsamples[j]) + '.pdf')

    ##############################################################################################
    plt.rc('lines', linewidth=linewidth[j])
    ##############################################################################################

    plt.figure(figsize=(22, 10))
    fig,ax=plt.subplots(1,1,figsize=(20,10))
    ax.xaxis.offsetText.set_fontsize(50)
    plt.tick_params(axis='x', labelsize=50)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(4,-4))
    plt.tick_params(axis='y', labelsize=50)
    plt.plot(g_orbit_vmec.time, g_orbit_vmec.r_pos, "r-", label="vmec")
    plt.plot(g_orbit_simple.time, g_orbit_simple.r_pos, "g--", label="simple")
    plt.legend(loc='lower right', fontsize=50)
    plt.xlabel(r"$t \ (s)$",fontsize=60,labelpad=20)
    plt.ylabel(r"s",fontsize=60,labelpad=20)
    plt.tight_layout()
    plt.savefig('NEAT/examples/misc/results/Radial_' + filename + str(nsamples[j]) + '.pdf')

    ##############################################################################################

    plt.figure(figsize=(10,10))
    plt.tick_params(axis='x', labelsize=50)
    plt.tick_params(axis='y', labelsize=50)
    plt.plot(
        g_orbit_vmec.rpos_cylindrical[0] * np.cos(g_orbit_vmec.rpos_cylindrical[2]),
        g_orbit_vmec.rpos_cylindrical[0] * np.sin(g_orbit_vmec.rpos_cylindrical[2]),
        "r-", label="vmec",
    )
    plt.plot(
       -g_orbit_simple.rpos_cylindrical[0] * np.cos(g_orbit_simple.rpos_cylindrical[2]),
        g_orbit_simple.rpos_cylindrical[0] * np.sin(g_orbit_simple.rpos_cylindrical[2]),
        "g--", label="simple",
    )
    plt.legend(loc='upper right',fontsize=50)
    plt.xlabel(r"$X \ (m)$",fontsize=60)
    plt.ylabel(r"$Y \ (m)$",fontsize=60)
    plt.tight_layout()
    plt.savefig('NEAT/examples/misc/results/Cyl_' + filename + str(nsamples[j]) + '.pdf')

    ##############################################################################################


    fig,ax1=plt.subplots(figsize=(10,10))
    circle=patches.Circle((0,0), radius=1,color='black',fill=False,linestyle='dotted',linewidth=2)
    circle2=patches.Circle((0,0), radius=r_initial,color='black',fill=False,linestyle='dotted',linewidth=2)
    ax1.add_patch(circle)
    ax1.add_patch(circle2)
    ax1.set(xlim=(-1.2,1.2),ylim=(-1.2,1.2))
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.plot(
        g_orbit_vmec.r_pos * np.cos(np.pi - g_orbit_vmec.theta_pos),
        g_orbit_vmec.r_pos * np.sin(np.pi - g_orbit_vmec.theta_pos),
        "r-", label="vmec",
    )
    plt.plot(
        g_orbit_simple.r_pos * np.cos(np.pi + g_orbit_simple.theta_pos),
        g_orbit_simple.r_pos * np.sin(np.pi + g_orbit_simple.theta_pos),
        "g--", label="simple",
    )
    plt.legend(loc='upper right',fontsize=50)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel(r"s cos($\theta$)",fontsize=60)
    plt.ylabel(r"s sin($\theta$)",fontsize=60)
    plt.savefig('results/Booz_' + filename + str(nsamples[j]) + '.pdf')

##############################################################################################

# if len(nsamples) > 1:
#         plt.figure(figsize=(10, 6))
#         plt.plot(nsamples, time_vmec, label="vmec")
#         plt.plot(nsamples, time_simple, label="simple")
#         plt.legend(loc='best',fontsize=16)
#         plt.xlabel("nsamples")
#         plt.ylabel("time (s)")
#         plt.savefig('results/Time_' + filename + str(nsamples[j]) + '.pdf')

