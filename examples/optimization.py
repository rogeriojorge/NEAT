#!/usr/bin/env python3

from neat.fields import stellna_qs
from neat.objectives import optimize_loss_fraction
from neat.tracing import charged_particle, charged_particle_ensemble, particle_orbit

r_surface_max = 0.06
r_initial = 0.05
energy = 1e3

g_field = stellna_qs(
    rc=[1, 0.2, 0.03, 0.004],
    zs=[0, 0.2, 0.03, 0.004],
    etabar=-1.5,
    nfp=4,
    B2c=0.4,
    order="r2",
    nphi=101,
)
g_particle = charged_particle_ensemble(r0=r_initial, energy=energy)
optimizer = optimize_loss_fraction(g_field, g_particle, r_surface_max=r_surface_max)
##################
print("Before run:")
print(" Iota: ", optimizer.field.iota)
print(" Max elongation: ", max(optimizer.field.elongation))
print(" Max Inverse L grad B: ", max(optimizer.field.inv_L_grad_B))
print(" Initial loss fraction: ", optimizer.loss_fraction.J())
print(" objective function: ", optimizer.prob.objective())
##################
optimizer.run(ftol=1e-3, nIterations=20)
##################
print("After run:")
print(" Iota: ", optimizer.field.iota)
print(" Max elongation: ", max(optimizer.field.elongation))
print(" Max Inverse L grad B: ", max(optimizer.field.inv_L_grad_B))
print(" Final loss fraction: ", optimizer.loss_fraction.J())
print(" objective function: ", optimizer.prob.objective())
print("        rc      = [", ",".join([str(elem) for elem in optimizer.field.rc]), "]")
print("        zs      = [", ",".join([str(elem) for elem in optimizer.field.zs]), "]")
print("        etabar = ", optimizer.field.etabar)
print("        B2c = ", optimizer.field.B2c)
##################
g_field_final = optimizer.field
g_particle_final = charged_particle(
    r0=r_initial, theta0=0, energy=energy, Lambda=1 / g_field_final.B0
)
g_orbit_final = particle_orbit(g_particle_final, g_field_final)
g_orbit_final.plot_orbit()
g_orbit_final.plot_orbit_3D()
