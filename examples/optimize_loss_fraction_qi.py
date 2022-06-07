#!/usr/bin/env python3

import glob
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from simsopt import LeastSquaresProblem, least_squares_serial_solve
from simsopt.solve.mpi import least_squares_mpi_solve
from simsopt.util.mpi import MpiPartition, log

from neat.fields import Stellna
from neat.objectives import EffectiveVelocityResidual, LossFractionResidual
from neat.tracing import ChargedParticle, ChargedParticleEnsemble, ParticleOrbit

r_initial = 0.04
r_max = 0.1
n_iterations = 10
ftol = 1e-5
B0 = 5
nsamples = 700
tfinal = 1e-4
stellarator_index = "QI NFP1 r2"
nthreads = 4
constant_b20 = False
energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
ntheta = 12  # resolution in theta
nphi = 6  # resolution in phi
nlambda_trapped = 18  # number of pitch angles for trapped particles
nlambda_passing = 3  # number of pitch angles for passing particles


class optimize_loss_fraction:
    def __init__(
        self,
        field,
        particles,
        r_max=r_max,
        nsamples=nsamples,
        tfinal=tfinal,
        nthreads=nthreads,
        parallel=False,
        constant_b20=constant_b20,
    ) -> None:

        self.field = field
        self.particles = particles
        self.nsamples = nsamples
        self.tfinal = tfinal
        self.nthreads = nthreads
        self.r_max = r_max
        self.parallel = parallel

        self.mpi = MpiPartition()

        # self.residual = LossFractionResidual(
        self.residual = EffectiveVelocityResidual(
            # LossFractionResidual(
            self.field,
            self.particles,
            self.nsamples,
            self.tfinal,
            self.nthreads,
            self.r_max,
            constant_b20=constant_b20,
        )

        self.field.fix_all()
        # self.field.unfix("etabar")
        # self.field.unfix("rc(1)")
        # self.field.unfix("zs(1)")
        self.field.unfix("rc(2)")
        # self.field.unfix("zs(2)")
        self.field.unfix("rc(3)")
        # self.field.unfix("zs(3)")
        # self.field.unfix("B2c")

        # Define objective function
        self.prob = LeastSquaresProblem.from_tuples(
            [
                (self.residual.J, 0, 40),
                # (self.field.get_elongation, 0.0, 2),
                (self.field.get_inv_L_grad_B, 0, 0.1),
                # (self.field.get_grad_grad_B_inverse_scale_length_vs_varphi, 0, 0.1),
                # (self.field.get_B20_mean, 0, 0.1),
            ]
        )

    def run(self, ftol=1e-6, n_iterations=100, rel_step=1e-3, abs_step=1e-5):
        # Algorithms that do not use derivatives
        # Relative/Absolute step size ~ 1/n_particles
        # with MPI, to see more info do mpi.write()
        if self.parallel:
            least_squares_mpi_solve(
                self.prob,
                self.mpi,
                grad=True,
                rel_step=rel_step,
                abs_step=abs_step,
                max_nfev=n_iterations,
            )
        else:
            least_squares_serial_solve(self.prob, ftol=ftol, max_nfev=n_iterations)


g_field_temp = Stellna.from_paper(stellarator_index, nphi=201)
g_field = Stellna.from_paper(
    stellarator_index, B0_vals=np.array(g_field_temp.B0_vals) * B0, nphi=201
)
g_particle = ChargedParticleEnsemble(
    r_initial=r_initial,
    r_max=r_max,
    energy=energy,
    charge=charge,
    mass=mass,
    ntheta=ntheta,
    nphi=nphi,
    nlambda_trapped=nlambda_trapped,
    nlambda_passing=nlambda_passing,
)
optimizer = optimize_loss_fraction(
    g_field,
    g_particle,
    r_max=r_max,
    tfinal=tfinal,
    nsamples=nsamples,
    constant_b20=constant_b20,
)
test_particle = ChargedParticle(
    r_initial=r_initial, theta_initial=np.pi / 2, phi_initial=np.pi, Lambda=0.97
)
##################
if optimizer.mpi.proc0_world:
    print("Before run:")
    print(" Iota: ", optimizer.field.iota)
    print(" Max elongation: ", max(optimizer.field.elongation))
    print(" Max Inverse L grad B: ", max(optimizer.field.inv_L_grad_B))
    # print(
    #     " Max Inverse L gradgrad B: ", optimizer.field.grad_grad_B_inverse_scale_length
    # )
    print(" Initial Mean residual: ", np.mean(optimizer.residual.J()))
    print(" Initial Loss Fraction: ", optimizer.residual.orbits.loss_fraction_array[-1])
    print(" Objective function: ", optimizer.prob.objective())
    print(" Initial equilibrium: ")
    print(
        "        rc      = [", ",".join([str(elem) for elem in optimizer.field.rc]), "]"
    )
    print(
        "        zs      = [", ",".join([str(elem) for elem in optimizer.field.zs]), "]"
    )
    print("        etabar = ", optimizer.field.etabar)
    # print("        B2c = ", optimizer.field.B2c)
    # print("        B20 = ", optimizer.field.B20_mean)
    optimizer.residual.orbits.plot_loss_fraction(show=False)
initial_orbit = ParticleOrbit(test_particle, g_field, nsamples=nsamples, tfinal=tfinal)
initial_field = Stellna.from_paper(
    stellarator_index, B0_vals=np.array(g_field_temp.B0_vals) * B0, nphi=201
)
##################
optimizer.run(ftol=ftol, n_iterations=n_iterations)
##################
if optimizer.mpi.proc0_world:
    print("After run:")
    print(" Iota: ", optimizer.field.iota)
    print(" Max elongation: ", max(optimizer.field.elongation))
    print(" Max Inverse L grad B: ", max(optimizer.field.inv_L_grad_B))
    # print(
    #     " Max Inverse L gradgrad B: ", optimizer.field.grad_grad_B_inverse_scale_length
    # )
    print(" Final Mean residual: ", np.mean(optimizer.residual.J()))
    print(" Final Loss Fraction: ", optimizer.residual.orbits.loss_fraction_array[-1])
    print(" Objective function: ", optimizer.prob.objective())
    print(" Final equilibrium: ")
    print(
        "        rc      = [", ",".join([str(elem) for elem in optimizer.field.rc]), "]"
    )
    print(
        "        zs      = [", ",".join([str(elem) for elem in optimizer.field.zs]), "]"
    )
    print("        etabar = ", optimizer.field.etabar)
    # print("        B2c = ", optimizer.field.B2c)
    # print("        B20 = ", optimizer.field.B20_mean)
    optimizer.residual.orbits.plot_loss_fraction(show=False)
    initial_patch = mpatches.Patch(color="#1f77b4", label="Initial")
    final_patch = mpatches.Patch(color="#ff7f0e", label="Final")
    plt.legend(handles=[initial_patch, final_patch])
final_orbit = ParticleOrbit(test_particle, g_field, nsamples=nsamples, tfinal=tfinal)
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
initial_orbit.plot_orbit_3d(show=False, r_surface=r_max)
final_orbit.plot_orbit_3d(show=False, r_surface=r_max)
# initial_orbit.plot_animation(show=False)
# final_orbit.plot_animation(show=True)
plt.show()

# Remove output files from simsopt
for f in glob.glob("residuals_202*"):
    os.remove(f)
for f in glob.glob("simsopt_202*"):
    os.remove(f)
