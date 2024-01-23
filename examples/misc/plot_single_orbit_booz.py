#!/usr/bin/env python
import time
from booz_xform import Booz_xform
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from scipy.interpolate import CubicSpline as spline

from neat.fields import Boozxform, StellnaQS
from neat.fields import Vmec as VMEC_NEAT, Simple
from neat.tracing import ChargedParticle, ParticleOrbit

"""                                                                           
Trace the orbit of a single particle in a
vmec equilibrium                
"""

# Initialize an alpha particle at a radius = r_initial
r_initial = 0.5  # initial normalized toroidal magnetic flux (radial VMEC coordinate)
theta_initial = 0.1   # initial poloidal angle
phi_initial = 0.1  # initial poloidal angle
energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
Lambda = np.array([0.1, 0.5, 0.9 ])  # = mu * B0 / energy
vpp_sign = -1  # initial sign of the parallel velocity, +1 or -1
nsamples = np.array([1000])  # resolution in time
tfinal = np.array([1e-4])  # seconds

B0 = 5.3267
Rmajor_ARIES = 7.7495
Rminor_ARIES = 1.7044

scalings=np.array([1.5,2,3])
Rmajors=scalings*Rmajor_ARIES
Aspect_ratios=np.round(Rmajors/Rminor_ARIES,2)
iterator=range(Aspect_ratios.size)
iterator1=range(Lambda.size)
iterator2=range(nsamples.size)

inputs_vmec = [f"input.nearaxis_{Aspect_ratios[0]}" for i in iterator]
filenames=[f"nearaxis_{Aspect_ratios[i]}_000_000000.nc" for i in iterator]
wout_filenames = ["wout_" + filenames[i] for i in iterator]
boozmn_filenames = ["boozmn_new_" + filenames[i] for i in iterator]

stellarator = ["precise QA", "2022 QH nfp4 well"]
g_field_basis = StellnaQS.from_paper(stellarator[1], B0=B0, nphi=101)
for i in iterator:
    g_field_qsc = StellnaQS(
        rc=g_field_basis.rc * Rmajors[i],
        zs=g_field_basis.zs * Rmajors[i],
        etabar=g_field_basis.etabar / Rmajors[i],
        B2c=g_field_basis.B2c * (B0 / Rmajors[i] / Rmajors[i]),
        B0=B0,
        nfp=g_field_basis.nfp,
        order="r3",
        nphi=101,
    )
    # Creating wout of VMEC
        # g_field.to_vmec(filename=inputs_vmec[i],r=Rminor_ARIES, params={"ntor":10, "mpol":10, \
        #   "niter_array":[10000,10000,20000],'ftol_array':[1e-13,1e-16,1e-19],'ns_array':[16,49,101]},
        #       ntheta=30, ntorMax=30) #standard ntheta=20, ntorMax=14
        # vmec=Vmec(filename=filename_vmec, verbose=True)
        # vmec.run()

    # Creating wout of Boozxform
    # b = Booz_xform()
    # b.read_wout(wout_filenames[i])
    # # b.comput_surfs=100
    # b.mboz = 100
    # b.nboz = 50
    # b.run()
    # b.write_boozmn(boozmn_filenames[i])

    g_field_vmec = VMEC_NEAT(wout_filename=wout_filenames[i], maximum_s=1)
    g_field_booz = Boozxform(wout_filename=boozmn_filenames[i])
    g_field_simple = Simple(wout_filename=wout_filenames[i], ns_s=5, ns_tp=5, multharm=4)
    for k in iterator1:
        g_particle = ChargedParticle(
            r_initial=r_initial,
            theta_initial=theta_initial,
            phi_initial=phi_initial,
            energy=energy,
            Lambda=Lambda[k],
            charge=charge,
            mass=mass,
            vpp_sign=vpp_sign,
        )

        # Transforming varphi into phi0 and consequently to phi_VMEC   
        nu_array = g_field_qsc.varphi - g_field_qsc.phi
        nu_spline_of_varphi = spline(
            np.append(g_field_qsc.varphi, 2 * np.pi / g_field_qsc.nfp),
            np.append(nu_array, nu_array[0]),
            bc_type="periodic",
        )                     
        phi0 = phi_initial - nu_spline_of_varphi(phi_initial)
        phi_VMEC=g_field_qsc.to_RZ([[Rminor_ARIES*np.sqrt(r_initial),theta_initial,phi0]])[2][0]

        time_vmec = []
        time_booz = []
        time_qsc = []
        time_simple = []
        for j in iterator2:
            print("=" * 80)
            print(f"nsamples = {nsamples[j]}")
            print("  Starting particle tracer vmec")
            start_time = time.time()
            g_particle.theta_initial = np.pi - theta_initial
            g_particle.phi_initial=phi_VMEC
            g_orbit_vmec = ParticleOrbit(
                g_particle, g_field_vmec, nsamples=nsamples[j], tfinal=tfinal[j]
            )
            total_time = time.time() - start_time
            print(f"  Finished vmec in {total_time}s")
            time_vmec.append(total_time)
            
            print("  Starting particle tracer simple")
            start_time = time.time()
            g_particle.theta_initial = -(np.pi - theta_initial)
            g_particle.phi_initial=-phi_VMEC
            g_orbit_simple = ParticleOrbit(
                g_particle, g_field_simple, nsamples=nsamples[j], tfinal=tfinal[j]
            )
            total_time = time.time() - start_time
            print(f"  Finished simple in {total_time}s")
            time_simple.append(total_time)

            g_particle.theta_initial = theta_initial
            g_particle.phi_initial=phi_initial

            print("  Starting particle tracer booz")
            start_time = time.time()
            g_orbit_booz = ParticleOrbit(
                g_particle, g_field_booz, nsamples=nsamples[j], tfinal=tfinal[j]
            )
            total_time = time.time() - start_time
            print(f"  Finished booz in {total_time}s")
            time_booz.append(total_time)

            print("  Starting particle tracer qsc")
            start_time = time.time()
            g_particle.r_initial = Rminor_ARIES * np.sqrt(r_initial)
            g_particle.theta_initial=theta_initial-(g_field_qsc.iota-g_field_qsc.iotaN)*phi_initial
            g_orbit_qsc = ParticleOrbit(
                g_particle, g_field_qsc, nsamples=nsamples[j], tfinal=tfinal[j], constant_b20=False
            )
            g_particle.r_initial = r_initial
            total_time = time.time() - start_time
            print(f"  Finished in {total_time}s")
            time_qsc.append(total_time)

            
            plt.figure(figsize=(10, 6))
            plt.subplot(2, 3, 1)
            plt.plot(g_orbit_vmec.time, np.pi - g_orbit_vmec.theta_pos, label="vmec")
            plt.plot(g_orbit_simple.time, np.pi + g_orbit_simple.theta_pos, label="simple")
            plt.plot(g_orbit_booz.time, g_orbit_booz.theta_pos-(g_field_qsc.iota-g_field_qsc.iotaN)*g_orbit_booz.varphi_pos, label="booz")
            plt.plot(g_orbit_qsc.time, g_orbit_qsc.theta_pos, "k--", label="qsc")
            plt.legend()
            plt.xlabel(r"$t (s)$")
            plt.ylabel(r"$\theta$")
            plt.subplot(2, 3, 2)
            plt.plot(g_orbit_vmec.time, g_orbit_vmec.varphi_pos, label="vmec")
            plt.plot(g_orbit_simple.time, -g_orbit_simple.varphi_pos, label="simple")
            plt.plot(g_orbit_booz.time, g_orbit_booz.varphi_pos, label="booz")
            plt.plot(g_orbit_qsc.time, g_orbit_qsc.varphi_pos, "k--", label="qsc")
            plt.legend()
            plt.xlabel(r"$t (s)$")
            plt.ylabel(r"$\varphi$")
            plt.subplot(2, 3, 3)
            plt.plot(g_orbit_vmec.time, g_orbit_vmec.v_parallel, label="vmec")
            plt.plot(g_orbit_simple.time, -g_orbit_simple.v_parallel, label="simple")
            plt.plot(g_orbit_booz.time, g_orbit_booz.v_parallel, label="booz")
            plt.plot(g_orbit_qsc.time, g_orbit_qsc.v_parallel, "k--", label="qsc")
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
            plt.plot(
                g_orbit_booz.time,
                (g_orbit_booz.total_energy - g_orbit_booz.total_energy[0])
                / g_orbit_booz.total_energy[0],
                label="booz",
            )
            plt.plot(
                g_orbit_qsc.time,
                (g_orbit_qsc.total_energy - g_orbit_qsc.total_energy[0])
                / g_orbit_qsc.total_energy[0],
                "k--",
                label="qsc",
            )
            plt.legend()
            plt.xlabel(r"$t (s)$")
            plt.ylabel(r"$(E-E_0)/E_0$")
            plt.subplot(2, 3, 5)
            plt.plot(g_orbit_vmec.time, g_orbit_vmec.rdot, "r-", label=r"$\dot r$ vmec")
            plt.plot(g_orbit_booz.time, g_orbit_booz.rdot, "b--", label=r"$\dot r$ booz")
            plt.plot(g_orbit_qsc.time, g_orbit_qsc.rdot, "g--", label=r"$\dot r$ qsc")
            plt.plot(g_orbit_vmec.time, -g_orbit_vmec.thetadot, "g-", label=r"$\dot \theta$ vmec")
            plt.plot(g_orbit_booz.time, g_orbit_booz.thetadot, "k--", label=r"$\dot \theta$ booz")
            plt.plot(g_orbit_qsc.time, g_orbit_qsc.thetadot, "y--", label=r"$\dot \theta$ qsc")
            plt.plot(g_orbit_vmec.time, g_orbit_vmec.varphidot, "m-", label=r"$\dot \varphi$ vmec")
            plt.plot(g_orbit_booz.time, g_orbit_booz.varphidot, "c--", label=r"$\dot \varphi$ booz")
            plt.plot(g_orbit_qsc.time, g_orbit_qsc.varphidot, "k--", label=r"$\dot \varphi$ qsc")
            plt.xlabel(r"$t (s)$")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.subplots_adjust(right=0.8)  # Adjust the right margin to fit the legend
            plt.subplot(2, 3, 6)
            plt.plot(g_orbit_vmec.time, g_orbit_vmec.magnetic_field_strength, label="vmec")
            # plt.plot(g_orbit_simple.time, g_orbit_simple.magnetic_field_strength, label="simple") #We do not get this value
            plt.plot(g_orbit_booz.time, g_orbit_booz.magnetic_field_strength, label="booz")
            plt.plot(g_orbit_qsc.time, g_orbit_qsc.magnetic_field_strength, "k--", label="qsc")
            plt.legend()
            plt.xlabel(r"$t$")
            plt.ylabel(r"$|B|$")
            plt.tight_layout()
            # plt.show()
            plt.savefig(f'results_local/L={Lambda[k]}_Stuff_{filenames[i]}_{nsamples[j]}.pdf')
            plt.close()

            ##############################################################################################
            # Create a color map
            plt.rc('lines', linewidth=3)
            # plt.rcParams['xtick.major.pad']=32
            ##############################################################################################
            # from matplotlib import ticker
            # plt.figure(figsize=(22, 10))
            fig,ax=plt.subplots(1,1,figsize=(20,10))
            # formatter=ticker.ScalarFormatter(useMathText=False)
            # formatter.set_scientific('True')
            # formatter.set_powerlimits((-4,4))
            # ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.offsetText.set_fontsize(50)
            plt.tick_params(axis='x', labelsize=50)
            plt.ticklabel_format(axis='x', style='sci', scilimits=(4,-4))
            plt.tick_params(axis='y', labelsize=50)
            plt.plot(g_orbit_vmec.time, g_orbit_vmec.r_pos, "r-", label="vmec")
            plt.plot(g_orbit_simple.time, g_orbit_simple.r_pos, "g--", label="simple")
            plt.plot(g_orbit_booz.time, g_orbit_booz.r_pos, "c-.", label="booz")
            plt.plot(g_orbit_qsc.time, (g_orbit_qsc.r_pos / (Rminor_ARIES)) ** 2, "k:", label="qsc")
            plt.legend(loc='lower right', fontsize=50)
            plt.xlabel(r"$t \ (s)$",fontsize=60,labelpad=20)
            plt.ylabel(r"s",fontsize=60,labelpad=20)
            plt.tight_layout()
            plt.savefig(f'results_local/L={Lambda[k]}_Radial_{filenames[i]}_{nsamples[j]}.pdf')

            ##############################################################################################

            plt.figure(figsize=(22,22))
            plt.tick_params(axis='x', labelsize=60)
            plt.tick_params(axis='y', labelsize=60)
            plt.plot(
                g_orbit_vmec.rpos_cylindrical[0]*np.cos(g_orbit_vmec.rpos_cylindrical[2]),
                g_orbit_vmec.rpos_cylindrical[0]*np.sin(g_orbit_vmec.rpos_cylindrical[2]),
                "r-", label="vmec",
            )
            # plt.plot(
            #     # # g_orbit_simple.rpos_cylindrical[0] * np.cos(-g_orbit_simple.rpos_cylindrical[2]),
            #     # # g_orbit_simple.rpos_cylindrical[0] * np.sin(-g_orbit_simple.rpos_cylindrical[2]),
            #     # "g--", label="simple",
            # )
            # try:
            #     coordinates=np.array([Rminor_ARIES * np.sqrt(g_orbit_booz.r_pos), g_orbit_booz.theta_pos, \
            #                                     g_orbit_booz.rpos_cylindrical[2]])
            #     coordinates_trans=np.einsum("ij->ji",coordinates)

            #     phi_cyl_booz = g_field_qsc.to_RZ(coordinates_trans)[2]
            
            #     plt.plot(
            #         g_orbit_booz.rpos_cylindrical[0]*np.cos(g_orbit_booz.rpos_cylindrical[2]),
            #         g_orbit_booz.rpos_cylindrical[0]*np.sin(g_orbit_booz.rpos_cylindrical[2]),
            #         "c-.", label="booz",
            #     )
            # except:
            #     print('No cyl booz')

            plt.plot(
                g_orbit_qsc.rpos_cylindrical[0]*np.cos(g_orbit_qsc.rpos_cylindrical[2]),
                g_orbit_qsc.rpos_cylindrical[0]*np.sin(g_orbit_qsc.rpos_cylindrical[2]),
                "k:",
                label="qsc",
            )
            plt.legend(loc='best',fontsize=50)
            plt.xlabel(r"$X \ (m)$",fontsize=60,labelpad=20)
            plt.ylabel(r"$Y \ (m)$",fontsize=60,labelpad=20)
            plt.tight_layout()
            plt.savefig(f'results_local/L={Lambda[k]}_Cyl_{filenames[i]}_{nsamples[j]}.pdf')
            ##############################################################################################

            fig,ax1=plt.subplots(figsize=(22,22))
            circle=patches.Circle((0,0), radius=1,color='black',fill=False,linestyle='dotted',linewidth=2)
            circle2=patches.Circle((0,0), radius=r_initial,color='black',fill=False,linestyle='dotted',linewidth=2)
            ax1.add_patch(circle)
            ax1.add_patch(circle2)
            ax1.set(xlim=(-1.2,1.2),ylim=(-1.2,1.2))

            plt.tick_params(axis='x', labelsize=60)
            plt.tick_params(axis='y', labelsize=60)
            # plt.plot(
            #     g_orbit_vmec.r_pos * np.cos(np.pi - g_orbit_vmec.theta_pos),
            #     g_orbit_vmec.r_pos * np.sin(np.pi - g_orbit_vmec.theta_pos),
            #     "r-", label="vmec",
            # )
            # plt.plot(
            #     g_orbit_simple.r_pos * np.cos(np.pi + g_orbit_simple.theta_pos),
            #     g_orbit_simple.r_pos * np.sin(np.pi + g_orbit_simple.theta_pos),
            #     "g--", label="simple",
            # )
            plt.plot(
                g_orbit_booz.r_pos * np.cos(g_orbit_booz.theta_pos-(g_field_qsc.iota-g_field_qsc.iotaN)*g_orbit_booz.varphi_pos),
                g_orbit_booz.r_pos * np.sin(g_orbit_booz.theta_pos-(g_field_qsc.iota-g_field_qsc.iotaN)*g_orbit_booz.varphi_pos),
                "c-.", label="booz",
            )
            plt.plot(
                (g_orbit_qsc.r_pos / (Rminor_ARIES)) ** 2 * np.cos(g_orbit_qsc.theta_pos),
                (g_orbit_qsc.r_pos / (Rminor_ARIES)) ** 2 * np.sin(g_orbit_qsc.theta_pos),
                "k:",
                label="qsc",
            )

            plt.legend(loc='best',fontsize=50)
            plt.gca().set_aspect("equal", adjustable="box")
            plt.xlabel(r"s cos($\theta$)",fontsize=60,labelpad=20)
            plt.ylabel(r"s sin($\theta$)",fontsize=60,labelpad=20)
            plt.tight_layout()
            plt.savefig(f'results_local/L={Lambda[k]}_Booz_{filenames[i]}_{nsamples[j]}.pdf')
            # plt.show()
            plt.close()

        # if len(nsamples) > 1:
        #         plt.figure(figsize=(10, 6))
        #         plt.plot(nsamples, time_vmec, label="vmec")
        #         # # plt.plot(nsamples, time_simple, label="simple")
        #         plt.plot(nsamples, time_booz, label="booz")
        #         plt.plot(nsamples, time_qsc, label="qsc")
        #         plt.legend(loc='best',fontsize=26)
        #         plt.xlabel("nsamples")
        #         plt.ylabel("time (s)")
        #         plt.savefig(f'results_local/L={Lambda[k]}_Time_{filenames[i]}_{nsamples[j]}.pdf')
