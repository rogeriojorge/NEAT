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

from desc.equilibrium import Equilibrium
from desc.vmec import VMECIO
from desc.objectives import get_fixed_boundary_constraints, get_NAE_constraints
from simsopt.mhd import Vmec
from neat.fields import Simple, StellnaQS, Vmec as Vmec_NEAT, Boozxform
from neat.tracing import ChargedParticle, ParticleOrbit
from neat.plotting import butter_lowpass_filter, butter_lowpass_filter2

# nfp=4
step_i=0.25

s_initials = np.round(np.arange(0.25,0.9,step_i), 2) # psi/psi_a for vmec
# s_initials = [0.1, 0.25,0.5, 0.75]

# Initial values for lambdas

lambdas=np.round(np.arange(0.3,0.89,0.2), 2) # = mu * B0 / energy
# lambdas=np.array([0.2])
# lambdas=np.array([0.9,0.95])

# Initial angular values

# theta_initials=np.round(np.arange(0,2*np.pi,2*np.pi/5), 2)         # initial poloidal angle (vartheta=theta_Boozer - N phi_Boozer)
# phi_initials=np.round(np.arange(0,2*np.pi/nfp,2*np.pi/nfp/4), 2)    # initial toroidal angle (cylindrical on axis)

theta_initials=[1.26]#,1.1266087039264279,2.0149839496633657, 3.1415926535897931]
phi_initials=[0.31]#,1.1266087039264279,2.0149839496633657, 3.1415926535897931]

# QA -> (theta, phi)= (0,0), (3.77,0)
# QH -> (theta, phi)= (0,0.94), (0.79,0.3), (1.26,0.93)

# Initializing lists
g_orbits=[]
g_orbits_vmec=[]
g_orbits_desc=[]
g_orbits_simple=[]
g_orbits_booz=[]

#Particle values
B0 = 5.3267         # Tesla, magnetic field on-axis (ARIES-CS)
energy = 3.52e6     # electron-volt
charge = 2          # times charge of proton
mass = 4            # times mass of proton     
vpp_sign = 1        # initial sign of the parallel velocity, +1 or -1
nsamples = 5000     # resolution in time
tfinal = 1e-4       # seconds
constant_b20 = False # use a constant B20 (mean value) or the real function

# Scaling values
Rmajor_ARIES = 7.7495*2  # Major radius double double of ARIES-CS
Rminor_ARIES = 1.7044  # Minor radius
Aspect=np.round(Rmajor_ARIES/Rminor_ARIES,2)

# Names of input from NA and of VMEC and DESC outputs
# filename_vmec = f"input.nearaxis_{Aspect}"
wout_filename_desc = f"wout_nearaxis_{Aspect}_desc.nc"
wout_filename_vmec = f"wout_nearaxis_{Aspect}_000_000000.nc"
boozmn_filename = f"boozmn_new_nearaxis_{Aspect}_000_000000.nc" 

# List of working stellarators
stellarator = ["precise QA", "2022 QH nfp4 well"]

# Initializing and scaling NA field
g_field_basis = StellnaQS.from_paper(stellarator[1], B0=B0, nphi=401)
g_field = StellnaQS(rc=g_field_basis.rc*Rmajor_ARIES, zs=g_field_basis.zs*Rmajor_ARIES, \
                    etabar=g_field_basis.etabar/Rmajor_ARIES, B2c=g_field_basis.B2c*(B0/Rmajor_ARIES/Rmajor_ARIES),\
                        B0=B0, nfp=g_field_basis.nfp, order='r3', nphi=401)

# Calculating variable nu that maps from varphi to phi0
nu_array = g_field.varphi - g_field.phi
nu_spline_of_varphi = spline(
    np.append(g_field.varphi, 2 * np.pi / g_field.nfp),
    np.append(nu_array, nu_array[0]),
    bc_type="periodic",
)

# Specs for DESC equilibrium from near-axis
field_desc = Equilibrium.from_near_axis(g_field, r=Rminor_ARIES, M=14, N=6)
# constraints = get_NAE_constraints(field_desc, g_field, iota=False, order=1)
# # constraints = get_fixed_boundary_constraints(iota=False)
# field_desc.solve(verbose=3, ftol=1e-2, objective="force", maxiter=100, xtol=1e-6, constraints=constraints)
# VMECIO.save(field_desc, wout_filename_desc)

# Creating wout of VMEC
# g_field.to_vmec(filename=filename_vmec,r=Rminor_ARIES, params={"ntor":10, "mpol":10, \
#   "niter_array":[10000,10000,20000],'ftol_array':[1e-13,1e-16,1e-19],'ns_array':[16,49,101]},
#       ntheta=30, ntorMax=30) #standard ntheta=20, ntorMax=14
# vmec=Vmec(filename=filename_vmec, verbose=True)
# vmec.run()

# Initializing all fields with wout from VMEC
# g_field_desc = Vmec_NEAT(wout_filename=wout_filename_desc,maximum_s=1)
g_field_vmec = Vmec_NEAT(wout_filename=wout_filename_vmec,maximum_s=1)
g_field_simple = Simple(wout_filename=wout_filename_vmec, ns_s=3, ns_tp=3, multharm=3,nsamples=nsamples, integmode=1)
g_field_booz = Boozxform(wout_filename=boozmn_filename)

for i in np.arange(lambdas.size):
    for s_initial in s_initials:
        for theta_initial in theta_initials:
            for phi_initial in phi_initials:
                for vpp_sign in [-1]:
                    
                    # Initializing particles 
                    g_particle = ChargedParticle(
                        r_initial = Rminor_ARIES*np.sqrt(s_initial),#s=np.sqrt(2*r_initial*psi_a/B0) bc psi_a=(B0*0.1*0.1)/2
                        theta_initial = theta_initial-(g_field.iota-g_field.iotaN)*phi_initial,
                        phi_initial = phi_initial,    
                        energy = energy,
                        Lambda = lambdas[i],
                        charge = charge,
                        mass = mass,
                        vpp_sign = vpp_sign,
                    )

                    # Transforming varphi into phi0 and consequently to phi_VMEC                        
                    phi0 = phi_initial - nu_spline_of_varphi(phi_initial)
                    phi_VMEC=g_field.to_RZ([[Rminor_ARIES*np.sqrt(s_initial),theta_initial,phi0]])[2][0]

                    # g_particle_desc = ChargedParticle(
                    #     r_initial = s_initial,
                    #     theta_initial =  np.pi + theta_initial,     
                    #     phi_initial = phi_VMEC,    
                    #     energy = energy,
                    #     Lambda = lambdas[i],
                    #     charge = charge,
                    #     mass = mass,
                    #     vpp_sign = -vpp_sign,
                    # )
                    g_particle_vmec = ChargedParticle(
                        r_initial = s_initial,
                        theta_initial = np.pi-(theta_initial),     
                        phi_initial = phi_VMEC,    
                        energy = energy,
                        Lambda = lambdas[i],
                        charge = charge,
                        mass = mass,
                        vpp_sign = vpp_sign,
                    )
                    g_particle_simple = ChargedParticle(
                        r_initial=s_initial,
                        theta_initial= -(np.pi-theta_initial),
                        phi_initial= -phi_VMEC,
                        energy=energy,
                        Lambda=lambdas[i],
                        charge=charge,
                        mass=mass,
                        vpp_sign= vpp_sign,
                    )

                    g_particle_booz = ChargedParticle(
                        r_initial=s_initial,
                        theta_initial = theta_initial,     
                        phi_initial = phi_initial,
                        energy=energy,
                        Lambda=lambdas[i],
                        charge=charge,
                        mass=mass,
                        vpp_sign=vpp_sign,
                    )

                    # Running particle tracers for chosen particles and fields
                    g_orbit = ParticleOrbit(g_particle, g_field, nsamples = nsamples, 
                        tfinal = tfinal, constant_b20 = constant_b20)
                    # g_orbit_desc = ParticleOrbit(g_particle_desc, g_field_desc, nsamples = nsamples,
                    #     tfinal = tfinal, constant_b20 = constant_b20)
                    g_orbit_vmec = ParticleOrbit(g_particle_vmec, g_field_vmec, nsamples = nsamples,
                        tfinal = tfinal, constant_b20 = constant_b20)
                    g_orbit_simple = ParticleOrbit(g_particle_simple, g_field_simple, nsamples=nsamples, 
                        tfinal=tfinal)
                    g_orbit_booz = ParticleOrbit(g_particle_booz, g_field_booz, nsamples=nsamples, 
                        tfinal=tfinal)
                    
                    # Correcting signs in NEAT to match others
                    g_orbit.theta_pos  = -g_orbit.theta_pos
                    g_orbit.thetadot   = -g_orbit.thetadot

                    #Correcting signs in SIMPLE to match others
                    g_orbit_simple.theta_pos  = -g_orbit_simple.theta_pos
                    g_orbit_simple.varphi_pos = -g_orbit_simple.varphi_pos
                    g_orbit_simple.thetadot   = -g_orbit_simple.thetadot
                    g_orbit_simple.varphidot  = -g_orbit_simple.varphidot
                    g_orbit_simple.rpos_cylindrical[1] = -g_orbit_simple.rpos_cylindrical[1]
                    g_orbit_simple.rpos_cylindrical[2] = -g_orbit_simple.rpos_cylindrical[2]
                    g_orbit_simple.rpos_cartesian[1]= -g_orbit_simple.rpos_cartesian[1]
                    g_orbit_simple.time = g_orbit_simple.time - g_orbit_simple.time[0]

                    # #Interpolating variables in SIMPLE to match times
                    g_orbit_simple.rpos_cylindrical= interp1d(g_orbit_simple.time, g_orbit_simple.rpos_cylindrical, 
                    assume_sorted=True, bounds_error=False, fill_value='extrapolate')(g_orbit.time)
                    g_orbit_simple.rpos_cartesian= interp1d(g_orbit_simple.time, g_orbit_simple.rpos_cartesian, 
                    assume_sorted=True, bounds_error=False, fill_value='extrapolate')(g_orbit.time)
                    g_orbit_simple.r_pos = interp1d(g_orbit_simple.time, g_orbit_simple.r_pos, 
                    assume_sorted=True, bounds_error=False, fill_value='extrapolate')(g_orbit.time)
                    g_orbit_simple.theta_pos=interp1d(g_orbit_simple.time, g_orbit_simple.theta_pos, 
                    assume_sorted=True, bounds_error=False, fill_value='extrapolate')(g_orbit.time)
                    g_orbit_simple.varphi_pos=interp1d(g_orbit_simple.time, g_orbit_simple.varphi_pos, 
                    assume_sorted=True, bounds_error=False, fill_value='extrapolate')(g_orbit.time)
                    g_orbit_simple.rdot = interp1d(g_orbit_simple.time, g_orbit_simple.rdot, 
                    assume_sorted=True, bounds_error=False, fill_value='extrapolate')(g_orbit.time)
                    g_orbit_simple.thetadot=interp1d(g_orbit_simple.time, g_orbit_simple.thetadot, 
                    assume_sorted=True, bounds_error=False, fill_value='extrapolate')(g_orbit.time)
                    g_orbit_simple.varphidot=interp1d(g_orbit_simple.time, g_orbit_simple.varphidot, 
                    assume_sorted=True, bounds_error=False, fill_value='extrapolate')(g_orbit.time)
                    g_orbit_simple.energy_parallel = interp1d(g_orbit_simple.time, g_orbit_simple.energy_parallel, 
                    assume_sorted=True, bounds_error=False, fill_value='extrapolate')(g_orbit.time)
                    g_orbit_simple.energy_perpendicular = interp1d(g_orbit_simple.time, g_orbit_simple.energy_perpendicular, 
                    assume_sorted=True, bounds_error=False, fill_value='extrapolate')(g_orbit.time)
                    g_orbit_simple.total_energy =g_orbit_simple.energy_parallel + g_orbit_simple.energy_perpendicular
                    g_orbit_simple.magnetic_field_strength = interp1d(g_orbit_simple.time, g_orbit_simple.magnetic_field_strength, 
                    assume_sorted=True, bounds_error=False, fill_value='extrapolate')(g_orbit.time)
                    g_orbit_simple.v_parallel = interp1d(g_orbit_simple.time, g_orbit_simple.v_parallel, 
                    assume_sorted=True, bounds_error=False, fill_value='extrapolate')(g_orbit.time)
                    g_orbit_simple.vparalleldot = interp1d(g_orbit_simple.time, g_orbit_simple.vparalleldot, 
                    assume_sorted=True, bounds_error=False, fill_value='extrapolate')(g_orbit.time)
                    g_orbit_simple.p_phi= interp1d(g_orbit_simple.time, g_orbit_simple.p_phi, 
                    assume_sorted=True, bounds_error=False, fill_value='extrapolate')(g_orbit.time)

                    g_orbit_simple.time = g_orbit.time
                    
                    # Appending orbits to orbit list
                    g_orbits.append(g_orbit)
                    g_orbits_vmec.append(g_orbit_vmec)
                    # g_orbits_desc.append(g_orbit_desc)
                    g_orbits_simple.append(g_orbit_simple)
                    g_orbits_booz.append(g_orbit_booz)

# plt.rcParams['axes.prop_cycle'] = cycler(color=['g', 'black', 'r'])
plt.style.use('seaborn-v0_8-white')
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams["figure.facecolor"] = "w"
plt.rcParams['lines.linewidth'] = 5
plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24)
plt.rc('font', size=24)
plt.rc('legend', fontsize=18)
plt.rc('lines', linewidth=5)

folder=stellarator[0] + '_A=' + str(Aspect) + '_Booz'
if not os.path.exists(folder):
    os.mkdir(folder)
norms_r_pos=[]
norms_r_pos_filt=[]
for i in np.arange(0,int(len(g_orbits)/(len(s_initials))),1):  
    plt.figure(figsize=(20, 12))
    
    plt.tick_params(axis='x', labelsize=50)
    plt.tick_params(axis='y', labelsize=50)
    plt.rcParams["figure.facecolor"] = "w"
    
    for j in np.arange(0, len(s_initials),1):
        for k in np.arange(0,1,2):
            
            s_i=i*len(s_initials)+j+k
            
            norm_r_pos = (g_orbits[s_i].r_pos/(Rminor_ARIES))**2

            if j+k==0:
                plt.plot(g_orbits_vmec[s_i].time*1e6, g_orbits_vmec[s_i].r_pos, 'r-',label='VMEC')
                # plt.plot(g_orbits_desc[s_i].time*1e6, g_orbits_desc[s_i].r_pos,'b-.',label='DESC')
                plt.plot(g_orbits_simple[s_i].time*1e6, g_orbits_simple[s_i].r_pos,'g--',label='SIMPLE')
                plt.plot(g_orbits_booz[s_i].time*1e6, g_orbits_booz[s_i].r_pos,'c-.',label='BOOZ')
                plt.plot(g_orbits[s_i].time*1e6, norm_r_pos, ls='dotted', c='k',label='pyQSC')
            else:
                plt.plot(g_orbits_vmec[s_i].time*1e6, g_orbits_vmec[s_i].r_pos, 'r-')
                plt.plot(g_orbits_simple[s_i].time*1e6, g_orbits_simple[s_i].r_pos, 'g--')
                plt.plot(g_orbits_booz[s_i].time*1e6, g_orbits_booz[s_i].r_pos,'c-.')
                # plt.plot(g_orbits_desc[s_i].time*1e6, g_orbits_desc[s_i].r_pos,'b-.')
                plt.plot(g_orbits[s_i].time*1e6, norm_r_pos, ls='dotted', c='k')
    if not os.path.exists(folder + '/' + str(i)):
        os.mkdir(folder + '/' + str(i))
    # plt.legend(fontsize='small', loc=(0.6,0.15), ncol=2)
    plt.tick_params(axis='x', labelsize=50,pad=20)
    plt.tick_params(axis='y', labelsize=50,pad=20)
    
    plt.legend(loc='lower right', fontsize=40)
    plt.xlabel(r't ($\mu$s)',fontsize=60,labelpad=20)
    plt.ylabel(r's',fontsize=60,labelpad=20)#$\psi$/$\psi_0$')
    plt.tight_layout()
    plt.ylim(0, 1)
    t=1000
    # plt.xlim(0*t,1*t)
    # plt.ylim(0.65,0.9)
    # leg=plt.legend()

    plt.savefig(folder + '/lambda=' + str(lambdas[i]) +  '_r_pos.pdf',transparent=False)
    # plt.savefig('best_try_' + str(s_i) + '.pdf')
    # plt.show()

    # plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams["figure.facecolor"] = "w"
# plt.rcParams['lines.linewidth'] = 5
plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24)
plt.rc('font', size=24)
plt.rc('legend', fontsize=18)
plt.rc('lines', linewidth=5)


if not os.path.exists(folder):
    os.mkdir(folder)
norms_r_pos=[]
norms_r_pos_filt=[]
# plt.figure(figsize=(10, 6))
for i in np.arange(0,len(g_orbits),1):  
    s_i=np.round(0.1 + step_i*i,2)
    if not os.path.exists(folder + '/' + str(s_i)):
        os.mkdir(folder + '/' + str(s_i))
    norm_r_pos = (g_orbits[i].r_pos/(Rminor_ARIES))**2
    
    plt.figure(figsize=(20, 12))
    plt.tick_params(axis='x', labelsize=50)
    plt.tick_params(axis='y', labelsize=50)
    plt.rcParams["figure.facecolor"] = "w"
    plt.plot(g_orbits_vmec[i].time*1e6, g_orbits_vmec[i].r_pos,'k-',label='VMEC')
    # plt.plot(g_orbits_desc[i].time*1e6, g_orbits_desc[i].r_pos, 'b-.',label='DESC')
    plt.plot(g_orbits_simple[i].time*1e6, g_orbits_simple[i].r_pos, 'g--',label='SIMPLE')
    plt.plot(g_orbits_booz[i].time*1e6, g_orbits_booz[i].r_pos,'c-.',label='BOOZ')
    plt.plot(g_orbits[i].time*1e6, norm_r_pos,'r',label='pyQSC', linestyle='dotted')
    plt.legend(loc='upper right', fontsize=50)
    plt.xlabel(r't ($\mu$s)',fontsize=60,labelpad=20)
    plt.ylabel(r's',fontsize=60,labelpad=20)#$\psi$/$\psi_0$')
    plt.tight_layout()
    plt.savefig(folder + '/' + str(s_i)+'/r_pos_BOOZ',transparent=False)
    plt.close()
    # plt.show()
    
    # Contour Prints
    print('NEAT')
    g_orbits[i].plot_orbit_contourB(savefig=folder + '/' + str(s_i)+'/B_neat', show=False)
    print('VMEC')
    g_orbits_vmec[i].plot_orbit_contourB(savefig=folder + '/' + str(s_i)+'/B_vmec', show=False)
    # print('DESC')
    # g_orbits_desc[i].plot_orbit_contourB(savefig=folder + '/' + str(s_i)+'/B_desc', show=False)
    print('SIMPLE')
    g_orbits_simple[i].plot_orbit_contourB(savefig=folder + '/' + str(s_i)+'/B_simple', show=False)
    print('BOOZ')
    g_orbits_booz[i].plot_orbit_contourB(savefig=folder + '/' + str(s_i)+'/B_booz', show=False)


    #Boozer coordinates comparisons
    print('NEAT-VMEC')
    g_orbits[i].plot_diff_boozer(g_orbits_vmec[i],r_minor=Rminor_ARIES,savefig=folder + '/' + str(s_i)+'/diff_booz_neat_vmec', show=False)
    # print('NEAT-DESC')
    # g_orbits[i].plot_diff_boozer(g_orbits_desc[i],r_minor=Rminor_ARIES)
    print('NEAT-SIMPLE')
    g_orbits[i].plot_diff_boozer(g_orbits_simple[i],r_minor=Rminor_ARIES,savefig=folder + '/' + str(s_i)+'/diff_booz_neat_simple', show=False)
    print('VMEC-SIMPLE')
    g_orbits_vmec[i].plot_diff_boozer(g_orbits_simple[i],r_minor=1,savefig=folder + '/' + str(s_i)+'/diff_booz_vmec_simple', show=False)
    print('NEAT-BOOZ')
    g_orbits[i].plot_diff_boozer(g_orbits_booz[i],r_minor=Rminor_ARIES,savefig=folder + '/' + str(s_i)+'/diff_booz_neat_booz', show=False)
    print('VMEC-BOOZ')
    g_orbits_vmec[i].plot_diff_boozer(g_orbits_booz[i],r_minor=Rminor_ARIES,savefig=folder + '/' + str(s_i)+'/diff_booz_vmec_booz', show=False)
    print('SIMPLE-BOOZ')
    g_orbits_simple[i].plot_diff_boozer(g_orbits_booz[i],r_minor=Rminor_ARIES,savefig=folder + '/' + str(s_i)+'/diff_booz_simple_booz', show=False)
    
    #Cylindrical coordinates comparisons
    print('NEAT-VMEC')
    g_orbits[i].plot_diff_cyl(g_orbits_vmec[i],savefig=folder + '/' + str(s_i)+'/diff_cyl_neat_vmec', show=False)
    # print('NEAT-DESC')
    # g_orbits[i].plot_diff_cyl(g_orbits_desc[i],savefig=folder + '/' + str(s_i)+'/diff_cyl_neat_desc', show=False)
    print('NEAT-SIMPLE')
    g_orbits[i].plot_diff_cyl(g_orbits_simple[i],savefig=folder + '/' + str(s_i)+'/diff_cyl_neat_simple', show=False)
    print('VMEC-SIMPLE')
    g_orbits_vmec[i].plot_diff_cyl(g_orbits_simple[i],savefig=folder + '/' + str(s_i)+'/diff_cyl_vmec_simple', show=False)
    print('NEAT-BOOZ')
    g_orbits[i].plot_diff_cyl(g_orbits_booz[i],savefig=folder + '/' + str(s_i)+'/diff_cyl_neat_booz', show=False)
    print('VMEC-BOOZ')
    g_orbits_vmec[i].plot_diff_cyl(g_orbits_booz[i],savefig=folder + '/' + str(s_i)+'/diff_cyl_vmec_booz', show=False)
    print('SIMPLE-BOOZ')
    g_orbits_simple[i].plot_diff_cyl(g_orbits_booz[i],savefig=folder + '/' + str(s_i)+'/diff_cyl_simple_booz', show=False)
    
    
    print('NEAT')
    g_orbits[i].plot(r_minor=Rminor_ARIES, savefig=folder + '/' + str(s_i)+'/param_neat', show=False)
    print('VMEC')
    g_orbits_vmec[i].plot(savefig=folder + '/' + str(s_i)+'/param_vmec', show=False)
    # print('DESC')
    # g_orbits_desc[i].plot(savefig=folder + '/' + str(s_i)+'/param_desc',  show=False)
    print('SIMPLE')
    g_orbits_simple[i].plot(savefig=folder + '/' + str(s_i)+'/param_simple', show=False)
    print('BOOZ')
    g_orbits_booz[i].plot(savefig=folder + '/' + str(s_i)+'/param_booz', show=False)
    
    # plt.legend(fontsize='small', loc=(0.6,0.15), ncol=2)
    # g_orbits[i].plot_orbit_3d(show=True, r_surface=Rminor_ARIES, savefig=folder + '/lambda=' + str(lambdas[1]) + '_s_i=' + str(s_initial) + \
    #     '_theta=' + str(theta_initial) + '_phi=' + str(phi_initial) + '_neat_orbit.png')
    # g_orbits_vmec[i].plot_orbit_3d(show=True,savefig=folder + '/lambda=' + str(lambdas[1]) + '_s_i=' + str(s_initial) + \
    #     '_theta=' + str(theta_initial) + '_phi=' + str(phi_initial) + '_vmec_orbit.png')
    # g_orbits_desc[i].plot_orbit_3d(show=True,savefig=folder + '/lambda=' + str(lambdas[1]) + '_s_i=' + str(s_initial) + \
    #     '_theta=' + str(theta_initial) + '_phi=' + str(phi_initial) + '_vmec_orbit.png')
    # try:
    #     max=np.minimum(avg_time[-1],avg_time2[-1],avg_time3[-1])
    #     plt.xlim(0,max)
    # except:
    #     print('No radial oscillation')
    # g_orbits[i].plot_orbit_contourB(show=True, savefig=folder + '/lambda=' + str(lambdas[0]) + '_s_i=' + str(s_initial) + \
    #      '_theta=' + str(theta_initial) + '_phi=' + str(phi_initial) + '_neat_orbit.png')

# plt.xlabel(r't ($\mu$s)')
# plt.ylabel(r"Radial oscillation amplitude")
# plt.legend(fontsize='small', loc=(0.6,0.15), ncol=2)
# # plt.ylabel(r'Average radial difference')
# # plt.legend(fontsize='small', loc=(0.6,0.1),ncol=2)
# # plt.ylabel(r"s")
# # plt.legend(fontsize='small', loc=(0.6,0.15), ncol=2)
# plt.show()



