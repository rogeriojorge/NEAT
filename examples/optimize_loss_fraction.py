#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from neat.fields import stellna_qs
from neat.objectives import optimize_loss_fraction
from neat.tracing import charged_particle, charged_particle_ensemble, particle_orbit

r_surface_max = 0.1
r_initial = 0.09
energy = 1e4

g_field = stellna_qs.from_paper(4, nphi=131)
g_particle = charged_particle_ensemble(r0=r_initial, energy=energy)
optimizer = optimize_loss_fraction(g_field, g_particle, r_surface_max=r_surface_max)
test_particle = charged_particle(r0=r_initial, theta0=np.pi, energy=energy, Lambda=1)
##################
print("Before run:")
print(" Iota: ", optimizer.field.iota)
print(" Max elongation: ", max(optimizer.field.elongation))
print(" Max Inverse L grad B: ", max(optimizer.field.inv_L_grad_B))
print(" Initial loss fraction: ", optimizer.loss_fraction.J())
print(" Objective function: ", optimizer.prob.objective())
print(" Initial equilibrium: ")
print("        rc      = [", ",".join([str(elem) for elem in optimizer.field.rc]), "]")
print("        zs      = [", ",".join([str(elem) for elem in optimizer.field.zs]), "]")
print("        etabar = ", optimizer.field.etabar)
print("        B2c = ", optimizer.field.B2c)
initial_orbit = particle_orbit(test_particle, g_field, nsamples=1000, Tfinal=0.0003)
##################
optimizer.run(ftol=1e-4, nIterations=20)
##################
print("After run:")
print(" Iota: ", optimizer.field.iota)
print(" Max elongation: ", max(optimizer.field.elongation))
print(" Max Inverse L grad B: ", max(optimizer.field.inv_L_grad_B))
print(" Final loss fraction: ", optimizer.loss_fraction.J())
print(" Objective function: ", optimizer.prob.objective())
print(" Final equilibrium: ")
print("        rc      = [", ",".join([str(elem) for elem in optimizer.field.rc]), "]")
print("        zs      = [", ",".join([str(elem) for elem in optimizer.field.zs]), "]")
print("        etabar = ", optimizer.field.etabar)
print("        B2c = ", optimizer.field.B2c)
final_orbit = particle_orbit(test_particle, g_field, nsamples=1000, Tfinal=0.0003)
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
plt.show()
