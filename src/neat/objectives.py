from qsc import Qsc
from simsopt._core.optimizable import Optimizable

import numpy as np

from neat.tracing import particle_ensemble_orbit


class loss_fraction_residual(Optimizable):
    def __init__(
        self,
        field: Qsc,
        particles: particle_ensemble_orbit,
        nsamples=500,
        Tfinal=0.0003,
        nthreads=8,
        r_max=0.12,
    ) -> None:

        self.field = field
        self.particles = particles
        self.nsamples = nsamples
        self.Tfinal = Tfinal
        self.nthreads = nthreads
        self.r_max = r_max

        Optimizable.__init__(self, depends_on=[field])

    def compute(self):
        self.orbits = particle_ensemble_orbit(
            self.particles, self.field, self.nsamples, self.Tfinal, self.nthreads
        )
        self.orbits.loss_fraction(r_max=self.r_max)

    def J(self):
        self.compute()
        return self.orbits.loss_fraction_array[-1]

class effective_velocity_residual(Optimizable):
    def __init__(
        self,
        field: Qsc,
        particles: particle_ensemble_orbit,
        nsamples=500,
        Tfinal=0.0003,
        nthreads=8,
        r_max=0.12,
    ) -> None:

        self.field = field
        self.particles = particles
        self.nsamples = nsamples
        self.Tfinal = Tfinal
        self.nthreads = nthreads
        self.r_max = r_max

        Optimizable.__init__(self, depends_on=[field])

    def compute(self):
        self.orbits = particle_ensemble_orbit(
            self.particles, self.field, self.nsamples, self.Tfinal, self.nthreads
        )
        self.orbits.loss_fraction(r_max=self.r_max)

        def radial_pos_of_particles(i, particle_pos):
            if self.orbits.lost_times_of_particles[i]==0:
                return max(particle_pos)
            else:
                return particle_pos[np.argmax(particle_pos > self.r_max)]

        time_of_particles = np.array([max(self.orbits.time) if self.orbits.lost_times_of_particles[i]==0 else self.orbits.lost_times_of_particles[i] for i in range(self.orbits.nparticles)])

        maximum_radial_pos_of_particles = np.array([
            radial_pos_of_particles(i, particle_pos) for i, particle_pos in enumerate(self.orbits.r_pos)
        ])

        self.effective_velocity = maximum_radial_pos_of_particles / time_of_particles

    def J(self):
        # effective diffusion coefficient that is continuous rather than discrete
        # delta s = maximum radial distance travelled by each particle before coliding with the wall or reaching the end of the simulation
        # delta t = time until particle collided or until the end of simulation, depends on the particle
        # J = delta s/ delta t or delta s^2/delta t
        # average radial diffusion coefficient/radial velocity. Make it as smal as possible
        self.compute()
        return 3e-3*self.effective_velocity / self.orbits.nparticles
