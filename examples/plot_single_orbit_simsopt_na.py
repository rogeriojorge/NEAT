#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from qsc import Qsc
from simsopt.field.boozermagneticfield import BoozerAnalytic
from simsopt.field.tracing import (
    MaxToroidalFluxStoppingCriterion,
    MinToroidalFluxStoppingCriterion,
    ToroidalTransitStoppingCriterion,
    trace_particles_boozer,
)
from simsopt.util.constants import ELEMENTARY_CHARGE, ONE_EV, PROTON_MASS

from neat.fields import StellnaQS
from neat.plotting import plot_orbit3d
from neat.tracing import ChargedParticle, ParticleOrbit

stellarator_index = "r1 section 5.1"
r_max = 0.1
tmax = 1e-4
energy = 1000
mass = 1
charge = 1
B0 = 4
nsamples = 1000
Lambda = 0.0

r_initial = 0.05
theta_initial = 0.5
phi_initial = 0.1

stel = Qsc.from_paper(stellarator_index, B0=B0)
bsh = BoozerAnalytic(
    stel.etabar,
    stel.B0,
    stel.iota - stel.iotaN,
    stel.G0,
    stel.B0 * r_max * r_max / 2,
    stel.iota,
)

Ekin = energy * ONE_EV
m = mass * PROTON_MASS
q = charge * ELEMENTARY_CHARGE
stz_inits = np.array(
    [[stel.B0 * r_initial * r_initial / 2, theta_initial, phi_initial]]
)
vpar_inits = np.array([np.sqrt(2 * Ekin / m * (1 - Lambda))])

bsh.set_points(stz_inits)

gc_tys, gc_phi_hits = trace_particles_boozer(
    bsh,
    stz_inits,
    vpar_inits,
    tmax=tmax,
    mass=m,
    charge=q,
    Ekin=Ekin,
    zetas=[],
    mode="gc_vac",
    stopping_criteria=[
        MinToroidalFluxStoppingCriterion(1e-6),
        MaxToroidalFluxStoppingCriterion(0.99),
        ToroidalTransitStoppingCriterion(1000, True),
    ],
    tol=1e-12,
)

g_field = StellnaQS.from_paper(stellarator_index, B0=B0)
g_particle = ChargedParticle(
    r_initial=r_initial,
    theta_initial=theta_initial,
    phi_initial=phi_initial,
    energy=energy,
    Lambda=Lambda,
    charge=charge,
    mass=mass,
    vpp_sign=1,
)
g_orbit = ParticleOrbit(
    g_particle, g_field, nsamples=nsamples, tfinal=tmax, constant_b20=True
)


time_simsopt = gc_tys[0][:, 0]
time_neat = g_orbit.time
r_pos_simsopt = np.sqrt(2 * gc_tys[0][:, 1] / stel.B0)
r_pos_neat = g_orbit.r_pos
theta_pos_simsopt = gc_tys[0][:, 2]
theta_pos_neat = g_orbit.theta_pos  # -g_orbit.varphi_pos*(stel.iotaN-stel.iota)
varphi_pos_simsopt = gc_tys[0][:, 3]
varphi_pos_neat = g_orbit.varphi_pos
v_parallel_simsopt = gc_tys[0][:, 4]
v_parallel_neat = g_orbit.v_parallel

plt.subplot(2, 2, 1)
plt.plot(time_simsopt, r_pos_simsopt, label=r"simsopt")
plt.plot(time_neat, r_pos_neat, label=r"neat")
plt.xlabel(r"$t (s)$")
plt.ylabel(r"$r$")
plt.legend()
plt.subplot(2, 2, 2)
plt.plot(time_simsopt, theta_pos_simsopt, label=r"simsopt")
plt.plot(time_neat, theta_pos_neat, label=r"neat")
plt.xlabel(r"$t (s)$")
plt.ylabel(r"$\theta$")
plt.legend()
plt.subplot(2, 2, 3)
plt.plot(time_simsopt, varphi_pos_simsopt, label=r"simsopt")
plt.plot(time_neat, varphi_pos_neat, label=r"neat")
plt.xlabel(r"$t (s)$")
plt.ylabel(r"$\varphi$")
plt.legend()
plt.subplot(2, 2, 4)
plt.plot(time_simsopt, v_parallel_simsopt, label=r"simsopt")
plt.plot(time_neat, v_parallel_neat, label=r"neat")
plt.xlabel(r"$t (s)$")
plt.ylabel(r"$v_\parallel$")
plt.legend()

# Plot orbit NEAT
g_orbit.plot_orbit_3d(show=False)

# Plot orbit SIMSOPT
boundary = np.array(
    stel.get_boundary(
        r=r_max,
        nphi=110,
        ntheta=30,
        ntheta_fourier=16,
        mpol=8,
        ntor=15,
    )
)
rpos_cylindrical_simsopt = np.array(
    stel.to_RZ(
        np.array([r_pos_simsopt, theta_pos_simsopt, varphi_pos_simsopt]).transpose()
    )
)
rpos_cartesian_simsopt = np.array(
    [
        rpos_cylindrical_simsopt[0] * np.cos(rpos_cylindrical_simsopt[2]),
        rpos_cylindrical_simsopt[0] * np.sin(rpos_cylindrical_simsopt[2]),
        rpos_cylindrical_simsopt[1],
    ]
)
plot_orbit3d(
    boundary=boundary,
    rpos_cartesian=rpos_cartesian_simsopt,
    distance=6,
    show=False,
)

plt.show()
