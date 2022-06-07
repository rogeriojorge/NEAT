from typing import Union

import numpy as np
from qic import Qic
from qsc import Qsc
from simsopt import LeastSquaresProblem, least_squares_serial_solve
from simsopt._core.optimizable import Optimizable

from neat.tracing import ParticleEnsembleOrbit


class loss_fraction_residual(Optimizable):
    def __init__(
        self,
        field: Union[Qsc, Qic],
        particles: ParticleEnsembleOrbit,
        nsamples=500,
        tfinal=0.0003,
        nthreads=2,
        r_max=0.12,
    ) -> None:

        self.field = field
        self.particles = particles
        self.nsamples = nsamples
        self.tfinal = tfinal
        self.nthreads = nthreads
        self.r_max = r_max

        Optimizable.__init__(self, depends_on=[field])

    def compute(self):
        self.orbits = ParticleEnsembleOrbit(
            self.particles, self.field, self.nsamples, self.tfinal, self.nthreads
        )
        self.orbits.loss_fraction(r_max=self.r_max)

    def J(self):
        self.compute()
        return self.orbits.loss_fraction_array[-1]


class effective_velocity_residual(Optimizable):
    def __init__(
        self,
        field: Union[Qsc, Qic],
        particles: ParticleEnsembleOrbit,
        nsamples=500,
        tfinal=0.0003,
        nthreads=2,
        r_max=0.12,
        constant_b20=False,
    ) -> None:

        self.field = field
        self.particles = particles
        self.nsamples = nsamples
        self.tfinal = tfinal
        self.nthreads = nthreads
        self.r_max = r_max
        self.constant_b20 = constant_b20

        Optimizable.__init__(self, depends_on=[field])

    def compute(self):
        self.orbits = ParticleEnsembleOrbit(
            self.particles,
            self.field,
            self.nsamples,
            self.tfinal,
            self.nthreads,
            constant_b20=self.constant_b20,
        )
        self.orbits.loss_fraction(r_max=self.r_max)

        def radial_pos_of_particles(i, particle_pos):
            if self.orbits.lost_times_of_particles[i] == 0:
                return max(particle_pos)  # subtract min(particle_pos), particle_pos[0]?
            else:
                return (
                    self.r_max
                )  # a bit more accurate -> particle_pos[np.argmax(particle_pos > self.r_max)]

        maximum_radial_pos_of_particles = np.array(
            [
                radial_pos_of_particles(i, particle_pos)
                for i, particle_pos in enumerate(self.orbits.r_pos)
            ]
        )

        tfinal = max(self.orbits.time)
        time_of_particles = np.array(
            [
                tfinal
                if self.orbits.lost_times_of_particles[i] == 0
                else self.orbits.lost_times_of_particles[i]
                for i in range(self.orbits.nparticles)
            ]
        )

        self.effective_velocity = maximum_radial_pos_of_particles / time_of_particles

    def J(self):
        # effective diffusion coefficient that is continuous rather than discrete
        # delta s = maximum radial distance travelled by each particle before coliding with the wall or reaching the end of the simulation
        # delta t = time until particle collided or until the end of simulation, depends on the particle
        # J = delta s/ delta t or delta s^2/delta t
        # average radial diffusion coefficient/radial velocity. Make it as smal as possible
        self.compute()
        return 1e-5 * self.effective_velocity / np.sqrt(self.orbits.nparticles)


class optimize_loss_fraction_skeleton:
    def __init__(
        self,
        field,
        particles,
        r_max=0.12,
        nsamples=800,
        tfinal=0.0001,
        nthreads=2,
        parallel=False,
    ) -> None:

        # log(level=logging.DEBUG)

        self.field = field
        self.particles = particles
        self.nsamples = nsamples
        self.tfinal = tfinal
        self.nthreads = nthreads
        self.r_max = r_max

        self.residual = loss_fraction_residual(
            self.field,
            self.particles,
            self.nsamples,
            self.tfinal,
            self.nthreads,
            self.r_max,
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
                (self.residual.J, 0, 1),
                # (self.field.get_elongation, 0.0, 3),
                # (self.field.get_inv_L_grad_B, 0, 2),
                # (self.field.get_grad_grad_B_inverse_scale_length_vs_varphi, 0, 2),
                # (self.field.get_B20_mean, 0, 0.01),
            ]
        )

    def run(self, ftol=1e-6, nIterations=100):
        least_squares_serial_solve(self.prob, ftol=ftol, max_nfev=nIterations)
