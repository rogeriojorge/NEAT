import logging

from qsc import Qsc
from simsopt import LeastSquaresProblem, least_squares_serial_solve
from simsopt._core.optimizable import Optimizable
from simsopt.solve.mpi import least_squares_mpi_solve
from simsopt.util.mpi import MpiPartition, log

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

    # def J_mynick
    # effective diffusion coefficient that is continuous rather than discrete
    # delta s = maximum radial distance travelled by each particle before coliding with the wall or reaching the end of the simulation
    # delta t = time until particle collided or until the end of simulation, depends on the particle
    # J = delta s/ delta t or delta s^2/delta t
    # average radial diffusion coefficient/radial velocity. Make it as smal as possible

    # As we work in Boozer coordinates, not in spacial coordinates, we don't initialize particles
    # uniformly in cartesian coordinates, in real space. To alleviate that, each particle initialization
    # or the objective function for each particle can be weighted by the volume jacobian
    # Jacobian in Boozer coordinates = (G/B^2)(r_0,theta_0,phi_0), ((G-N*I)/B^2)(r_0,theta_0,phi_0) if theta is theta-N phi (check!)


class optimize_loss_fraction:
    def __init__(
        self,
        field,
        particles,
        r_max=0.12,
        nsamples=800,
        Tfinal=0.0001,
        nthreads=8,
        parallel=False,
    ) -> None:

        # log(level=logging.DEBUG)

        self.field = field
        self.particles = particles
        self.nsamples = nsamples
        self.Tfinal = Tfinal
        self.nthreads = nthreads
        self.r_max = r_max
        self.parallel = parallel

        self.mpi = MpiPartition()

        self.loss_fraction = loss_fraction_residual(
            self.field,
            self.particles,
            self.nsamples,
            self.Tfinal,
            self.nthreads,
            self.r_max,
        )

        self.field.fix_all()
        self.field.unfix("etabar")
        self.field.unfix("rc(1)")
        self.field.unfix("zs(1)")
        self.field.unfix("rc(2)")
        self.field.unfix("zs(2)")
        self.field.unfix("rc(3)")
        self.field.unfix("zs(3)")
        self.field.unfix("B2c")

        # Define objective function
        self.prob = LeastSquaresProblem.from_tuples(
            [
                (self.loss_fraction.J, 0, 1),
                # (self.field.get_elongation, 0.0, 0.1),
                # (self.field.get_inv_L_grad_B, 0, 0.1),
                # (self.field.get_grad_grad_B_inverse_scale_length_vs_varphi, 0, 0.1),
                # (self.field.get_B20_mean, 0, 0.05),
            ]
        )

    def run(self, ftol=1e-6, nIterations=100, rel_step=1e-3, abs_step=1e-5):
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
                max_nfev=nIterations,
            )
        else:
            least_squares_serial_solve(self.prob, ftol=ftol, max_nfev=nIterations)
