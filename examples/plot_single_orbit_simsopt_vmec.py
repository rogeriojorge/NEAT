#!/usr/bin/env python3

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from simsopt.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from simsopt.field.tracing import (
    IterationStoppingCriterion,
    MaxToroidalFluxStoppingCriterion,
    MinToroidalFluxStoppingCriterion,
    ToroidalTransitStoppingCriterion,
    compute_poloidal_transits,
    compute_resonances,
    compute_toroidal_transits,
    trace_particles_boozer,
)
from simsopt.mhd import Vmec as Vmec_SIMSOPT
from simsopt.util.constants import ELEMENTARY_CHARGE, ONE_EV, PROTON_MASS

from neat.fields import Vmec as Vmec_NEAT
from neat.plotting import get_vmec_boundary, plot_orbit3d
from neat.tracing import ChargedParticle, ParticleOrbit

logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')
logger.setLevel(1)

wout_filename = (
    f"{os.path.join(os.path.dirname(__file__))}/inputs/wout_W7X.nc"  # vmec output
)

energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
Lambda = 0.97  # = mu * B0 / energy
vpp_sign = -1  # initial sign of the parallel velocity, +1 or -1
nsamples = 2000  # resolution in time
tmax = 1e-5  # seconds

nparticles = 1
s_initial = 0.4
theta_initial = np.pi / 2
phi_initial = np.pi

Ekin = energy * ONE_EV
m = mass * PROTON_MASS
q = charge * ELEMENTARY_CHARGE


order = 3
from simsopt.mhd.boozer import Boozer
vmec = Vmec_SIMSOPT(wout_filename)
booz = Boozer(vmec, mpol=10, ntor=10)
bri = BoozerRadialInterpolant(booz, order, rescale=False,
                                ns_delete=1, mpol=10, ntor=10)
nfp = vmec.wout.nfp
degree = 3
srange = (0, 1, 15)
thetarange = (0, np.pi, 15)
zetarange = (0, 2 * np.pi / nfp, 15)
field = InterpolatedBoozerField(
    bri, degree, srange, thetarange, zetarange, True, nfp=nfp, stellsym=True
)
print("Error in |B| interpolation", field.estimate_error_modB(1000), flush=True)
stz_inits = np.array([[s_initial, theta_initial, phi_initial]])
vpar_inits = np.array([np.sqrt(2 * Ekin / m * (1 - Lambda))])
field.set_points(stz_inits)
modB_inits = field.modB()
print("Starting tracing particles boozer")
gc_tys, gc_zeta_hits = trace_particles_boozer(
    field,
    stz_inits,
    vpar_inits,
    tmax=1e-2,
    mass=mass,
    charge=ELEMENTARY_CHARGE,
    Ekin=Ekin,
    tol=1e-8,
    mode="gc_vac",
    stopping_criteria=[
        # MaxToroidalFluxStoppingCriterion(0.99),
        # MinToroidalFluxStoppingCriterion(0.01),
        # ToroidalTransitStoppingCriterion(100, True),
    ],
    forget_exact_path=False,
)
print("Finish tracing particles boozer")

g_field = Vmec_NEAT(wout_filename=wout_filename)
g_particle = ChargedParticle(
    r_initial=s_initial,
    theta_initial=theta_initial,
    phi_initial=phi_initial,
    energy=energy,
    Lambda=Lambda,
    charge=charge,
    mass=mass,
    vpp_sign=vpp_sign,
)
g_orbit = ParticleOrbit(
    g_particle, g_field, nsamples=nsamples, tfinal=tmax, constant_b20=True
)

time_simsopt = gc_tys[0][:, 0]
time_neat = g_orbit.time
r_pos_simsopt = gc_tys[0][:, 1]
r_pos_neat = g_orbit.r_pos
theta_pos_simsopt = gc_tys[0][:, 2]
theta_pos_neat = g_orbit.theta_pos
varphi_pos_simsopt = gc_tys[0][:, 3]
varphi_pos_neat = g_orbit.varphi_pos
v_parallel_simsopt = gc_tys[0][:, 4]
v_parallel_neat = g_orbit.v_parallel

plt.subplot(2, 2, 1)
plt.plot(time_simsopt, r_pos_simsopt, label=r"simsopt")
plt.plot(time_neat, r_pos_neat, label=r"neat")
plt.xlabel(r"$t (s)$")
plt.ylabel(r"$s$")
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

# # Plot orbit NEAT
# g_orbit.plot_orbit_3d(show=False)

# Plto 2D orbit

index = 0
index_2pi = np.argmin(np.abs(gc_tys[index][:, 3] + 2 * np.pi))
x_SIMSOPT = np.sqrt(gc_tys[index][0:index_2pi, 1]) * np.cos(
    gc_tys[index][0:index_2pi, 2]
)
y_SIMSOPT = np.sqrt(gc_tys[index][0:index_2pi, 1]) * np.sin(
    gc_tys[index][0:index_2pi, 2]
)
x_NEAT = (r_pos_neat * np.cos(theta_pos_neat),)
y_NEAT = (r_pos_neat * np.sin(theta_pos_neat),)

plt.figure()
plt.plot(x_SIMSOPT, y_SIMSOPT, marker="*", linestyle="none", label="SIMSOPT")
plt.plot(x_NEAT, y_NEAT, marker="*", linestyle="none", label="SIMSOPT")
plt.xlabel(r"$x = \sqrt{s} \cos(\theta)$")
plt.ylabel(r"$y = \sqrt{s} \sin(\theta)$")

plt.show()
