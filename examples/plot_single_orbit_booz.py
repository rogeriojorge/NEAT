#!/usr/bin/env python
import time

import matplotlib.pyplot as plt
import numpy as np

from neat.fields import Boozxform, StellnaQS
from neat.fields import Vmec as VMEC_NEAT
from neat.tracing import ChargedParticle, ParticleOrbit

"""                                                                           
Trace the orbit of a single particle in a
vmec equilibrium                
"""

# Initialize an alpha particle at a radius = r_initial
r_initial = 0.5  # initial normalized toroidal magnetic flux (radial VMEC coordinate)
theta_initial = np.pi/2  # initial poloidal angle
phi_initial = 0.1  # initial poloidal angle
energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
Lambda = 0.99  # = mu * B0 / energy
vpp_sign = -1  # initial sign of the parallel velocity, +1 or -1
nsamples_array = [3000]  # resolution in time
tfinal = 3e-4  # seconds

B0 = 5.3267
Rmajor_ARIES = 7.7495 * 2
Rminor_ARIES = 1.7044
r_avg = Rminor_ARIES
filename = f"nearaxis_sboundary{Rmajor_ARIES/r_avg}_TRY_000_000000.nc"
filename_vmec = f"input.nearaxis_sboundary{Rmajor_ARIES/r_avg}_TRY"
wout_filename = "wout_" + filename
boozmn_filename = "boozmn_new_" + filename

stellarator = "precise QA"
g_field_basis = StellnaQS.from_paper(stellarator, B0=B0, nphi=401)
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
g_field_vmec = VMEC_NEAT(wout_filename=wout_filename, maximum_s=1)
g_field_booz = Boozxform(wout_filename=boozmn_filename)

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
time_booz = []
time_qsc = []
for nsamples in nsamples_array:
    print("=" * 80)
    print(f"nsamples = {nsamples}")
    print("  Starting particle tracer vmec")
    start_time = time.time()
    g_particle.theta_initial = np.pi - theta_initial
    g_orbit_vmec = ParticleOrbit(
        g_particle, g_field_vmec, nsamples=nsamples, tfinal=tfinal
    )
    g_particle.theta_initial = theta_initial
    total_time = time.time() - start_time
    print(f"  Finished vmec in {total_time}s")
    time_vmec.append(total_time)

    print("  Starting particle tracer booz")
    start_time = time.time()
    g_orbit_booz = ParticleOrbit(
        g_particle, g_field_booz, nsamples=nsamples, tfinal=tfinal
    )
    total_time = time.time() - start_time
    print(f"  Finished booz in {total_time}s")
    time_booz.append(total_time)

    print("  Starting particle tracer qsc")
    start_time = time.time()
    g_particle.r_initial = r_avg * np.sqrt(r_initial)
    g_orbit_qsc = ParticleOrbit(
        g_particle, g_field_qsc, nsamples=nsamples, tfinal=tfinal, constant_b20=False
    )
    g_particle.r_initial = r_initial
    total_time = time.time() - start_time
    print(f"  Finished in {total_time}s")
    time_qsc.append(total_time)

if len(nsamples_array) > 1:
    plt.figure(figsize=(10, 6))
    plt.plot(nsamples_array, time_vmec, label="vmec")
    plt.plot(nsamples_array, time_booz, label="booz")
    plt.plot(nsamples_array, time_qsc, label="qsc")
    plt.legend()
    plt.xlabel("nsamples")
    plt.ylabel("time (s)")
    plt.show()

plt.figure(figsize=(10, 6))
plt.subplot(3, 3, 1)
plt.plot(g_orbit_vmec.time, g_orbit_vmec.r_pos, label="vmec")
plt.plot(g_orbit_booz.time, g_orbit_booz.r_pos, label="booz")
plt.plot(g_orbit_qsc.time, (g_orbit_qsc.r_pos / (r_avg)) ** 2, "k--", label="qsc")
plt.legend()
plt.xlabel(r"$t (s)$")
plt.ylabel(r"$r$")
plt.subplot(3, 3, 2)
plt.plot(g_orbit_vmec.time, np.pi - g_orbit_vmec.theta_pos, label="vmec")
plt.plot(g_orbit_booz.time, g_orbit_booz.theta_pos, label="booz")
plt.plot(g_orbit_qsc.time, g_orbit_qsc.theta_pos, "k--", label="qsc")
plt.legend()
plt.xlabel(r"$t (s)$")
plt.ylabel(r"$\theta$")
plt.subplot(3, 3, 3)
plt.plot(g_orbit_vmec.time, g_orbit_vmec.varphi_pos, label="vmec")
plt.plot(g_orbit_booz.time, g_orbit_booz.varphi_pos, label="booz")
plt.plot(g_orbit_qsc.time, g_orbit_qsc.varphi_pos, "k--", label="qsc")
plt.legend()
plt.xlabel(r"$t (s)$")
plt.ylabel(r"$\varphi$")
plt.subplot(3, 3, 4)
plt.plot(g_orbit_vmec.time, g_orbit_vmec.v_parallel, label="vmec")
plt.plot(g_orbit_booz.time, g_orbit_booz.v_parallel, label="vmec")
plt.plot(g_orbit_qsc.time, g_orbit_qsc.v_parallel, "k--", label="qsc")
plt.legend()
plt.xlabel(r"$t (s)$")
plt.ylabel(r"$v_\parallel$")
plt.subplot(3, 3, 5)
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
plt.subplot(3, 3, 6)
plt.plot(
    g_orbit_vmec.rpos_cylindrical[0],
    g_orbit_vmec.rpos_cylindrical[1],
    label="vmec",
)
plt.plot(
    g_orbit_booz.rpos_cylindrical[0],
    g_orbit_booz.rpos_cylindrical[1],
    label="booz",
)
plt.plot(
    g_orbit_qsc.rpos_cylindrical[0],
    g_orbit_qsc.rpos_cylindrical[1],
    "k--",
    label="qsc",
)
plt.legend()
plt.xlabel(r"$R$")
plt.ylabel(r"$Z$")
plt.subplot(3, 3, 7)
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
plt.subplot(3, 3, 8)
plt.plot(
    g_orbit_vmec.r_pos * np.cos(np.pi - g_orbit_vmec.theta_pos),
    g_orbit_vmec.r_pos * np.sin(np.pi - g_orbit_vmec.theta_pos),
    label="vmec",
)
plt.plot(
    g_orbit_booz.r_pos * np.cos(g_orbit_booz.theta_pos),
    g_orbit_booz.r_pos * np.sin(g_orbit_booz.theta_pos),
    label="booz",
)
plt.plot(
    (g_orbit_qsc.r_pos / (r_avg)) ** 2 * np.cos(g_orbit_qsc.theta_pos),
    (g_orbit_qsc.r_pos / (r_avg)) ** 2 * np.sin(g_orbit_qsc.theta_pos),
    "k--",
    label="qsc",
)
plt.legend()
plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel(r"r cos($\theta$)")
plt.ylabel(r"r sin($\theta$)")
plt.subplot(3, 3, 9)
plt.plot(g_orbit_vmec.time, g_orbit_vmec.magnetic_field_strength, label="vmec")
plt.plot(g_orbit_booz.time, g_orbit_booz.magnetic_field_strength, label="booz")
plt.plot(g_orbit_qsc.time, g_orbit_qsc.magnetic_field_strength, "k--", label="qsc")
plt.legend()
plt.xlabel(r"$t$")
plt.ylabel(r"$|B|$")
plt.tight_layout()
plt.show()
