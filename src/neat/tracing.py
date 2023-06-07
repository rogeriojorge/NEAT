""" Tracing module of NEAT

This script performs the tracing of particles by
connecting the user input to the gyronimo-based
functions defined in the neatpp.cpp file. It is
able to simulate both single particle and particle
ensemble orbits.

"""

try:
    from .fields import Simple
except ImportError as error:
    pass
try:
    from .fields import Stellna
except ImportError as error:
    pass
try:
    from .fields import StellnaQS
except ImportError as error:
    pass
try:
    from .fields import Dommaschk
except ImportError as error:
    pass

import logging
from typing import Union

# import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline as spline

from .constants import ELEMENTARY_CHARGE, PROTON_MASS

# from .plotting import plot_animation3d, plot_orbit2d, plot_orbit3d, plot_parameters

logger = logging.getLogger(__name__)


class ChargedParticle:
    r"""
    Class that contains the physics information of a
    given charged particle, as well as its position
    and velocity
    """

    def __init__(
        self,
        charge=2,
        mass=4,
        Lambda=1.0,
        vpp_sign=1,
        energy=3.52e6,
        r_initial=0.05,
        theta_initial=np.pi,
        phi_initial=0,
    ) -> None:

        self.charge = charge
        self.mass = mass
        self.energy = energy
        self.Lambda = Lambda
        self.vpp_sign = vpp_sign
        self.r_initial = r_initial
        self.theta_initial = theta_initial
        self.phi_initial = phi_initial

    def is_alpha_particle(self) -> bool:
        """Return true if particle is an alpha particle"""
        return bool(
            self.mass == 4
            and self.charge == 2
            and np.isclose(self.energy, 3.52e6, rtol=5e-2)
        )

    def gyronimo_parameters(self):
        """Return list of parameters to feed gyronimo-based functions"""
        return (
            self.charge,
            self.mass,
            self.Lambda,
            self.vpp_sign,
            self.energy,
            self.r_initial,
            self.theta_initial,
            self.phi_initial,
        )


class ChargedParticleEnsemble:
    r"""
    Class that contains the physics information of a
    collection of charged particles, as well as their position
    and velocities
    """

    def __init__(
        self,
        charge=2,
        mass=4,
        energy=3.52e6,
        nlambda_trapped=10,
        nlambda_passing=3,
        r_initial=0.05,
        r_max=0.1,
        ntheta=10,
        nphi=10,
    ) -> None:

        self.charge = charge
        self.mass = mass
        self.energy = energy
        self.nlambda_trapped = nlambda_trapped
        self.nlambda_passing = nlambda_passing
        self.r_initial = r_initial
        self.r_max = r_max
        self.ntheta = ntheta
        self.nphi = nphi

    def is_alpha_particle(self) -> bool:
        """Return true if particles are a collection of alpha particles"""
        return bool(
            self.mass == 4
            and self.charge == 2
            and np.isclose(self.energy, 3.52e6, rtol=5e-2)
        )

    def gyronimo_parameters(self):
        """Return list of parameters to feed gyronimo-based functions"""
        return (
            self.charge,
            self.mass,
            self.energy,
            self.nlambda_trapped,
            self.nlambda_passing,
            self.r_initial,
            self.r_max,
            self.ntheta,
            self.nphi,
        )


class ParticleOrbit:  # pylint: disable=R0902
    r"""
    Interface function with the C++ executable NEAT. Receives a
    particle and a field instance from NEAT.
    """

    def __init__(
        self,
        particle: ChargedParticle,
        field: Union[StellnaQS, Stellna],
        nsamples=1000,
        tfinal=0.0001,
        constant_b20=False,
    ) -> None:

        self.particle = particle
        self.field = field
        self.nsamples = nsamples
        self.tfinal = tfinal

        self.field.constant_b20 = constant_b20

        self.gyronimo_parameters = [
            *self.field.gyronimo_parameters(),
            *self.particle.gyronimo_parameters(),
            self.nsamples,
            self.tfinal,
        ]

        solution = np.array(
            self.field.neatpp_solver(
                *self.field.gyronimo_parameters(),
                *self.particle.gyronimo_parameters(),
                self.nsamples,
                self.tfinal,
            )
        )

        self.solution = solution

        self.time = solution[:, 0]
        self.r_pos = solution[:, 1]
        self.theta_pos = solution[:, 2]
        self.varphi_pos = solution[:, 3]
        if self.field.near_axis:
            nu_array = field.varphi - field.phi
            nu_spline_of_varphi = spline(
                np.append(field.varphi, 2 * np.pi / field.nfp),
                np.append(nu_array, nu_array[0]),
                bc_type="periodic",
            )
            self.phi_pos = self.varphi_pos - nu_spline_of_varphi(self.varphi_pos)
        self.energy_parallel = solution[:, 4]
        self.energy_perpendicular = solution[:, 5]
        self.total_energy = self.energy_parallel + self.energy_perpendicular

        self.magnetic_field_strength = solution[:, 6]
        self.v_parallel = solution[:, 7]
        self.rdot = solution[:, 8]
        self.thetadot = solution[:, 9]
        self.varphidot = solution[:, 10]
        self.vparalleldot = solution[:, 11]

        if self.field.near_axis:
            self.p_phi = canonical_angular_momentum(
                particle,
                field,
                self.r_pos,
                self.v_parallel,
                self.magnetic_field_strength,
            )

            self.rpos_cylindrical = np.array(
                self.field.to_RZ(
                    np.array([self.r_pos, self.theta_pos, self.varphi_pos]).transpose()
                )
            )

        else:
            # Canonical angular momentum still not calculated for VMEC fields yet
            self.p_phi = np.array([1e-16] * len(self.time))
            self.rpos_cylindrical = np.array(
                [solution[:, 12], solution[:, 14], solution[:, 13]]
            )

        self.rpos_cartesian = np.array(
            [
                self.rpos_cylindrical[0] * np.cos(self.rpos_cylindrical[2]),
                self.rpos_cylindrical[0] * np.sin(self.rpos_cylindrical[2]),
                self.rpos_cylindrical[1],
            ]
        )

    def plot_orbit(self, show=True):
        """Plot particle orbit in 2D flux coordinates"""
        from .plotting import plot_orbit2d

        plot_orbit2d(
            x_position=self.r_pos * np.cos(self.theta_pos),
            y_position=self.r_pos * np.sin(self.theta_pos),
            show=show,
        )

    def plot_orbit_3d(self, r_surface=0.1, distance=6, show=True):
        """Plot particle orbit in 3D cartesian coordinates"""
        from .plotting import get_vmec_boundary, plot_orbit3d

        if self.field.near_axis:
            boundary = np.array(
                self.field.get_boundary(
                    r=r_surface,
                    nphi=110,
                    ntheta=30,
                    ntheta_fourier=16,
                    mpol=8,
                    ntor=15,
                )
            )
        else:
            boundary, b_rescaled = get_vmec_boundary(  # pylint: disable=W0612
                self.field.wout_filename
            )

        plot_orbit3d(
            boundary=boundary,
            rpos_cartesian=self.rpos_cartesian,
            distance=distance,
            show=show,
        )

    def plot(self, show=True):
        """Plot relevant physics parameters of the particle orbit"""
        from .plotting import plot_parameters

        plot_parameters(self=self, show=show)

    def plot_animation(self, r_surface=0.1, distance=7, show=True, save_movie=False):
        """Plot three-dimensional animation of the particle orbit"""
        from .plotting import get_vmec_boundary, plot_animation3d

        if self.field.near_axis:
            boundary = np.array(
                self.field.get_boundary(
                    r=r_surface,
                    nphi=110,
                    ntheta=30,
                    ntheta_fourier=16,
                    mpol=8,
                    ntor=15,
                )
            )
        else:
            boundary, b_rescaled = get_vmec_boundary(  # pylint: disable=W0612
                self.field.wout_filename
            )

        plot_animation3d(
            boundary=boundary,
            rpos_cartesian=self.rpos_cartesian,
            nsamples=self.nsamples,
            distance=distance,
            show=show,
            save_movie=save_movie,
        )

    def plot_orbit_contourB(self, ntheta=100, nphi=120, ncontours=20, show=True):
        """Plot particle orbit superimposed in B contours"""
        import matplotlib.pyplot as plt

        from .plotting import get_vmec_magB

        theta_array = np.linspace(0, 2 * np.pi, ntheta)
        phi_array = np.linspace(0, 2 * np.pi, nphi)
        phi_2D, theta_2D = np.meshgrid(phi_array, theta_array)
        if self.field.near_axis:
            b_on_surface = self.field.B_mag(
                self.r_pos[0], theta_2D, phi_2D, Boozer_toroidal=True
            )
        else:
            b_on_surface = get_vmec_magB(
                wout_filename=self.field.wout_filename,
                spos=self.r_pos[0],
                ntheta=ntheta,
                nzeta=nphi,
            )
        fig, ax = plt.subplots()
        plt.contourf(phi_2D, theta_2D, b_on_surface, ncontours)
        # plt.title(titles[i]+'\n1-based index='+str(iradius+1))
        ax.scatter(
            np.mod(self.varphi_pos, 2 * np.pi),
            np.mod(self.theta_pos, 2 * np.pi),
            marker=".",
            color="k",
            s=0.7,
        )
        ax.scatter(
            np.mod(self.varphi_pos[0], 2 * np.pi),
            np.mod(self.theta_pos[0], 2 * np.pi),
            marker="o",
            color="b",
            s=60,
        )
        plt.xlabel(r"$\phi$")
        plt.ylabel(r"$\theta$")
        plt.colorbar()
        plt.xlim([0, 2 * np.pi])
        plt.ylim([0, 2 * np.pi])
        if show:
            plt.show()


class ParticleEnsembleOrbit:  # pylint: disable=R0902
    r"""
    Interface function with the C++ executable NEAT. Receives a
    particle and field instance from neat.
    """

    def __init__(
        self,
        particles: ChargedParticleEnsemble,
        field: Union[StellnaQS, Stellna],
        nsamples=800,
        tfinal=0.0001,
        nthreads=2,
        constant_b20=True,
    ) -> None:

        self.particles = particles
        self.field = field
        self.nsamples = nsamples
        self.nthreads = nthreads
        self.tfinal = tfinal

        self.field.constant_b20 = constant_b20

        self.gyronimo_parameters = [
            *self.field.gyronimo_parameters(),
            *self.particles.gyronimo_parameters(),
            self.nsamples,
            self.tfinal,
            self.nthreads,
        ]

        solution = np.array(
            self.field.neatpp_solver_ensemble(
                *self.field.gyronimo_parameters(),
                *self.particles.gyronimo_parameters(),
                self.nsamples,
                self.tfinal,
                self.nthreads,
            )
        )

        self.solution = solution

        self.time = solution[:, 0]
        self.nparticles = solution.shape[1] - 1
        self.r_pos = solution[:, 1:].transpose()

        # Save the values of theta, phi and lambda used
        r_max = self.particles.r_max
        self.B_max = (
            abs(np.max(field.B0))
            + r_max * (abs(np.max(field.B1c)) + abs(np.max(field.B1s)))
            + r_max
            * r_max
            * (
                abs(np.max(field.B20))
                + abs(np.max(field.B2c_array))
                + abs(np.max(field.B2s_array))
            )
        )
        self.B_min = max(
            0.01,
            abs(np.max(field.B0))
            - r_max * (abs(np.max(field.B1c)) + abs(np.max(field.B1s)))
            - r_max
            * r_max
            * (
                abs(np.max(field.B20))
                + abs(np.max(field.B2c_array))
                + abs(np.max(field.B2s_array))
            ),
        )
        self.theta = np.linspace(0.0, 2 * np.pi, particles.ntheta)
        self.phi = np.linspace(0.0, 2 * np.pi / field.nfp, particles.nphi)
        self.lambda_trapped = np.linspace(
            field.Bbar / self.B_max, field.Bbar / self.B_min, particles.nlambda_trapped
        )
        self.lambda_passing = np.linspace(
            0.0,
            field.Bbar / self.B_max * (1.0 - 1.0 / particles.nlambda_passing),
            particles.nlambda_passing,
        )
        self.lambda_all = np.concatenate([self.lambda_trapped, self.lambda_passing])

        # Store the initial values for each particle in a single ordered array
        self.initial_lambda_theta_phi_vppsign = []
        for Lambda in self.lambda_all:
            for theta in self.theta:
                for phi in self.phi:
                    self.initial_lambda_theta_phi_vppsign.append(
                        [Lambda, theta, phi, +1]
                    )
                    self.initial_lambda_theta_phi_vppsign.append(
                        [Lambda, theta, phi, -1]
                    )

        # Compute the Jacobian for each particle at time t=0
        ## Jacobian in (r, vartheta, varphi) coordinates is given by J = (G+iota*I)*r*Bbar/B^2
        self.initial_jacobian = []
        for initial_lambda_theta_phi_vppsign in self.initial_lambda_theta_phi_vppsign:
            Lambda = initial_lambda_theta_phi_vppsign[0]
            theta = initial_lambda_theta_phi_vppsign[1]
            phi = initial_lambda_theta_phi_vppsign[2]
            radius = particles.r_initial
            field_magnitude = field.B_mag(radius, theta, phi, Boozer_toroidal=True)
            self.initial_jacobian.append(
                (field.G0 + radius * radius * field.G2 + field.iota * field.I2)
                * radius
                * field.Bbar
                / field_magnitude
                / field_magnitude
            )

        self.lost_times_of_particles = []
        self.loss_fraction_array = []
        self.total_particles_lost = 0

    def loss_fraction(self, r_max=0.15, jacobian_weight=True):
        """
        Weight each particle by its Jacobian in order to attribute to each particle
        a marker representing a set of particles. The higher the value of the Jacobian
        at the initial time, the higher the number of particles should be there. For
        this reason, the loss_fraction is weighted by the Jacobian at initial time if
        the flag jacobian_weight is set to True.
        """
        self.lost_times_of_particles = [
            self.time[np.argmax(particle_pos > r_max)] for particle_pos in self.r_pos
        ]
        self.loss_fraction_array = [0.0]
        self.total_particles_lost = 0
        if jacobian_weight:
            for time in self.time[1:]:
                index_particles_lost = [
                    i for i, x in enumerate(self.lost_times_of_particles) if x == time
                ]
                initial_jacobian_particles_lost_at_this_time = (
                    [0]
                    if not index_particles_lost
                    else [self.initial_jacobian[i] for i in index_particles_lost]
                )
                self.total_particles_lost += np.sum(
                    initial_jacobian_particles_lost_at_this_time
                )
                self.loss_fraction_array.append(
                    self.total_particles_lost / np.sum(self.initial_jacobian)
                )
        else:
            for time in self.time[1:]:
                particles_lost_at_this_time = self.lost_times_of_particles.count(time)
                self.total_particles_lost += particles_lost_at_this_time
                self.loss_fraction_array.append(
                    self.total_particles_lost / self.nparticles
                )

        return self.loss_fraction_array

    def plot_loss_fraction(self, show=True, save=False):
        """Make a plot of the fraction of total particles lost over time"""
        import matplotlib.pyplot as plt

        plt.semilogx(self.time, self.loss_fraction_array)
        plt.xlabel("Time (s)")
        plt.ylabel("Loss Fraction")
        plt.tight_layout()
        if save:
            plt.savefig("plot_loss_fraction.pdf")
        if show:
            plt.show()


class ParticleEnsembleOrbit_Simple:  # pylint: disable=R0902
    r"""
    Interface function with the SIMPLE compiled Fortran functions.
    Receives a particle and field instance from neat.
    """

    def __init__(
        self,
        particles: ChargedParticleEnsemble,
        field: Union[StellnaQS, Stellna],
        nsamples=5000,
        tfinal=0.001,
        nthreads=2,
        nparticles=32,
        notrace_passing=0,
        npoiper=100,
        npoiper2=128,
        nper=1000,
    ) -> None:

        self.particles = particles
        # Change later to a definition of a variable called nparticles
        self.nparticles = nparticles
        self.particles.ntheta = nparticles
        self.particles.nphi = 1
        self.particles.nlambda_passing = 1
        self.particles.nlambda_trapped = 1
        self.field = field
        self.nsamples = nsamples
        self.nthreads = nthreads
        self.tfinal = tfinal
        self.notrace_passing = notrace_passing
        self.npoiper = npoiper
        self.npoiper2 = npoiper2
        self.nper = nper

        # self.field.constant_b20 = constant_b20

        self.gyronimo_parameters = [
            *self.field.gyronimo_parameters(),
            *self.particles.gyronimo_parameters(),
            self.nsamples,
            self.nparticles,
            self.tfinal,
            self.nthreads,
            self.notrace_passing,
            self.npoiper,
            self.npoiper2,
            self.nper,
        ]

        solution = np.array(
            self.field.neatpp_solver_ensemble(
                *self.field.gyronimo_parameters(),
                *self.particles.gyronimo_parameters(),
                self.nsamples,
                self.nparticles,
                self.tfinal,
                self.nthreads,
                self.notrace_passing,
                self.npoiper,
                self.npoiper2,
                self.nper,
            ),
            dtype=object,
        )

        (
            self.time,
            self.confpart_pass,
            self.confpart_trap,
            self.trace_time,
            self.lost_times_of_particles,
            self.perp_inv,
        ) = solution

        self.condi = np.logical_and(
            self.lost_times_of_particles > 0,
            self.lost_times_of_particles < self.trace_time,
        )

        self.loss_fraction_array = 1 - (self.confpart_pass + self.confpart_trap)
        self.total_particles_lost = self.loss_fraction_array[-1]

    def plot_loss_fraction(self, show=True, save=False):
        """Make a plot of the fraction of total particles lost over time"""

        import matplotlib.pyplot as plt

        plt.figure()
        plt.semilogx(self.time, 1 - (self.confpart_pass + self.confpart_trap))
        plt.xlim([1e-6, self.trace_time])
        plt.xlabel("Time (s)")
        plt.ylabel("Loss Fraction")
        plt.tight_layout()

        if save:
            plt.savefig("plot_loss_fraction.pdf")

        plt.figure()
        plt.semilogx(
            self.lost_times_of_particles[self.condi], self.perp_inv[self.condi], "x"
        )
        plt.xlim([1e-6, self.trace_time])
        plt.xlabel("Loss Time")
        plt.ylabel("Perpendicular Invariant")

        if save:
            plt.savefig("plot_perpendicular_invariant.pdf")

        if show:
            plt.show()

    def save_loss_fraction(self, filename: str):
        data = np.column_stack([self.time, self.loss_fraction_array])
        np.savetxt(filename, data, fmt=["%s", "%s"])


def canonical_angular_momentum(
    particle, field, r_pos, v_parallel, magnetic_field_strength
):
    """
    Calculate the canonical angular momentum conjugated with
    the Boozer coordinate phi. This should be constant for
    quasi-symmetric stellarators.
    """
    m_proton = PROTON_MASS
    elementary_charge = ELEMENTARY_CHARGE

    p_phi1 = (
        particle.mass
        * m_proton
        * v_parallel
        * (field.G0 + r_pos**2 * (field.G2 + (field.iota - field.iotaN) * field.I2))
        / magnetic_field_strength
        / field.Bbar
    )

    p_phi2 = (
        particle.charge * elementary_charge * r_pos**2 * field.Bbar / 2 * field.iotaN
    )
    p_phi = p_phi1 - p_phi2

    return p_phi
