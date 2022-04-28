from qsc import Qsc
from simsopt import LeastSquaresProblem, least_squares_serial_solve
from simsopt._core.optimizable import Optimizable

from neat.tracing import particle_ensemble_orbit


class loss_fraction_residual(Optimizable):
    def __init__(
        self,
        field: Qsc,
        particles: particle_ensemble_orbit,
        nthreads=8,
        r_surface_max=0.15,
    ) -> None:

        self.field = field
        self.particles = particles
        self.nthreads = nthreads
        self.r_surface_max = r_surface_max

        Optimizable.__init__(self, depends_on=[field])

    def compute(self):
        self.orbits = particle_ensemble_orbit(
            self.particles, self.field, nthreads=self.nthreads
        )
        self.orbits.loss_fraction(r_surface_max=self.r_surface_max)

    def J(self):
        self.compute()
        return 100 * self.orbits.loss_fraction_array[-1]


class optimize_loss_fraction:
    def __init__(self, field, particles, r_surface_max=0.15, nthreads=8) -> None:
        self.field = field
        self.particles = particles
        self.nthreads = nthreads
        self.r_surface_max = r_surface_max

        self.loss_fraction = loss_fraction_residual(
            self.field, self.particles, self.nthreads, self.r_surface_max
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
                (self.loss_fraction.J, 0, 10),
                (self.field.get_elongation, 0.0, 0.1),
                (self.field.get_inv_L_grad_B, 0, 0.1),
            ]
        )

    def run(self, ftol=1e-6, nIterations=100):
        least_squares_serial_solve(self.prob, ftol=ftol, max_nfev=nIterations)
