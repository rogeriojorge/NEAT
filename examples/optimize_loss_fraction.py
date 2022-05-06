#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

from neat.fields import stellna_qs
from neat.objectives import optimize_loss_fraction
from neat.tracing import charged_particle, charged_particle_ensemble, particle_orbit

r_initial = 0.07
r_max = 0.1
nIterations = 20
ftol=1e-5
B0 = 5
B2c = B0/8
nsamples=800
Tfinal=0.00003
stellarator_index = 2

g_field = stellna_qs.from_paper(stellarator_index, nphi=151, B2c=B2c, B0=B0)
g_particle = charged_particle_ensemble(r0=r_initial, r_max=r_max)
optimizer = optimize_loss_fraction(g_field, g_particle, r_max=r_max, Tfinal=Tfinal, nsamples=nsamples)
test_particle = charged_particle(r0=r_initial, theta0=np.pi, Lambda=1.0)
##################
if optimizer.mpi.proc0_world:
    print("Before run:")
    print(" Iota: ", optimizer.field.iota)
    print(" Max elongation: ", max(optimizer.field.elongation))
    print(" Max Inverse L grad B: ", max(optimizer.field.inv_L_grad_B))
    print(" Max Inverse L gradgrad B: ", optimizer.field.grad_grad_B_inverse_scale_length)
    print(" Initial loss fraction: ", optimizer.loss_fraction.J())
    print(" Objective function: ", optimizer.prob.objective())
    print(" Initial equilibrium: ")
    print("        rc      = [", ",".join([str(elem) for elem in optimizer.field.rc]), "]")
    print("        zs      = [", ",".join([str(elem) for elem in optimizer.field.zs]), "]")
    print("        etabar = ", optimizer.field.etabar)
    print("        B2c = ", optimizer.field.B2c)
    print("        B20 = ", optimizer.field.B20_mean)
    optimizer.loss_fraction.orbits.plot_loss_fraction(show=False)
initial_orbit = particle_orbit(test_particle, g_field, nsamples=nsamples, Tfinal=Tfinal)
initial_field = stellna_qs.from_paper(stellarator_index, nphi=151, B2c=B2c, B0=B0)
##################
optimizer.run(ftol=ftol, nIterations=nIterations)
##################
if optimizer.mpi.proc0_world:
    print("After run:")
    print(" Iota: ", optimizer.field.iota)
    print(" Max elongation: ", max(optimizer.field.elongation))
    print(" Max Inverse L grad B: ", max(optimizer.field.inv_L_grad_B))
    print(" Max Inverse L gradgrad B: ", optimizer.field.grad_grad_B_inverse_scale_length)
    print(" Final loss fraction: ", optimizer.loss_fraction.J())
    print(" Objective function: ", optimizer.prob.objective())
    print(" Final equilibrium: ")
    print("        rc      = [", ",".join([str(elem) for elem in optimizer.field.rc]), "]")
    print("        zs      = [", ",".join([str(elem) for elem in optimizer.field.zs]), "]")
    print("        etabar = ", optimizer.field.etabar)
    print("        B2c = ", optimizer.field.B2c)
    print("        B20 = ", optimizer.field.B20_mean)
    optimizer.loss_fraction.orbits.plot_loss_fraction(show=False)
    initial_patch = mpatches.Patch(color=u'#1f77b4', label='Initial')
    final_patch = mpatches.Patch(color=u'#ff7f0e', label='Final')
    plt.legend(handles=[initial_patch, final_patch])
final_orbit = particle_orbit(test_particle, g_field, nsamples=nsamples, Tfinal=Tfinal)
final_field = g_field
##################
plt.figure()
plt.plot(
    initial_orbit.r_pos * np.cos(initial_orbit.theta_pos),
    initial_orbit.r_pos * np.sin(initial_orbit.theta_pos),
    label="Initial Orbit",
)
plt.plot(
    final_orbit.r_pos * np.cos(final_orbit.theta_pos),
    final_orbit.r_pos * np.sin(final_orbit.theta_pos),
    label="Final Orbit",
)
plt.gca().set_aspect("equal", adjustable="box")
plt.legend()
plt.xlabel(r"r cos($\theta$)")
plt.ylabel(r"r sin($\theta$)")
plt.tight_layout()
initial_orbit.plot_orbit_3D(show=False)
initial_orbit.plot_orbit_3D(show=False)
# initial_orbit.plot_animation(show=False)
# final_orbit.plot_animation(show=True)
plt.show()