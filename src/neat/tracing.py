import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline as spline

from neatpp import gc_solver_qs, gc_solver_qs_ensemble

from .util.constants import ELEMENTARY_CHARGE, MU_0, PROTON_MASS
from .util.plotting import set_axes_equal

logger = logging.getLogger(__name__)


class charged_particle:
    r"""
    Class that contains the physics information of a
    given charged particle, as well as its position
    and velocity
    """

    def __init__(
        self,
        charge=1,
        rhom=1,
        mass=1,
        Lambda=1.0,
        energy=4e4,
        r0=0.05,
        theta0=np.pi,
        phi0=0,
    ) -> None:
        self.charge = charge
        self.rhom = rhom
        self.mass = mass
        self.energy = energy
        self.Lambda = Lambda
        self.r0 = r0
        self.theta0 = theta0
        self.phi0 = phi0

    def gyronimo_parameters(self):
        return (
            self.charge,
            self.rhom,
            self.mass,
            self.Lambda,
            self.energy,
            self.r0,
            self.theta0,
            self.phi0,
        )


class charged_particle_ensemble:
    r"""
    Class that contains the physics information of a
    collection of charged particles, as well as their position
    and velocities
    """

    def __init__(
        self,
        charge=1,
        rhom=1,
        mass=1,
        energy=4e4,
        r0=0.05,
        theta0=np.pi,
        phi0=0,
    ) -> None:
        self.charge = charge
        self.rhom = rhom
        self.mass = mass
        self.energy = energy
        self.r0 = r0
        self.theta0 = theta0
        self.phi0 = phi0

    def gyronimo_parameters(self):
        return (
            self.charge,
            self.rhom,
            self.mass,
            self.energy,
            self.r0,
            self.theta0,
            self.phi0,
        )


class particle_orbit:
    r"""
    Interface function with the C++ executable NEAT. Receives a pyQSC instance
    and outputs the characteristics of the orbit.
    Args:
        stel: Qsc instance of pyQSC
        params (dict): a Python dict() containing the following parameters:
            r0,theta0,phi0,charge,rhom,mass,Lambda,energy,nsamples,Tfinal
        B20real (bool): True if a constant B20real should be used, False otherwise
    """

    def __init__(self, particle, field, nsamples=500, Tfinal=600) -> None:

        self.particle = particle
        self.field = field
        self.nsamples = nsamples
        self.Tfinal = Tfinal

        solution = np.array(
            gc_solver_qs(
                *field.gyronimo_parameters(),
                *particle.gyronimo_parameters(),
                nsamples,
                Tfinal
            )
        )
        self.gyronimo_parameters = solution

        self.time = solution[:, 0]
        self.r_pos = solution[:, 1]
        self.theta_pos = solution[:, 2]
        self.varphi_pos = solution[:, 3]

        nu = field.varphi - field.phi
        nu_spline_of_varphi = spline(
            np.append(field.varphi, 2 * np.pi / field.nfp),
            np.append(nu, nu[0]),
            bc_type="periodic",
        )

        self.phi_pos = self.varphi_pos - nu_spline_of_varphi(self.varphi_pos)
        self.energy_parallel = solution[:, 4]
        self.energy_perpendicular = solution[:, 5]
        self.total_energy = self.energy_parallel + self.energy_perpendicular
        self.Bfield = solution[:, 6]
        self.v_parallel = solution[:, 7]
        self.rdot = solution[:, 8]
        self.thetadot = solution[:, 9]
        self.phidot = solution[:, 10]
        self.vppdot = solution[:, 11]

        self.p_phi = canonical_angular_momentum(
            particle, field, self.r_pos, self.v_parallel, self.Bfield
        )

    def plot_orbit(self):
        x = self.r_pos * np.cos(self.theta_pos)
        y = self.r_pos * np.sin(self.theta_pos)
        plt.plot(x, y)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel(r"r cos($\theta$)")
        plt.ylabel(r"r sin($\theta$)")
        plt.tight_layout()
        plt.show()

    def plot_orbit_3D(self, distance=6):
        points = np.array([self.r_pos, self.theta_pos, self.varphi_pos]).transpose()
        rpos_cylindrical = np.array(self.field.to_RZ(points))
        rpos_cartesian = [
            rpos_cylindrical[0] * np.cos(rpos_cylindrical[2]),
            rpos_cylindrical[0] * np.sin(rpos_cylindrical[2]),
            rpos_cylindrical[1],
        ]
        boundary = np.array(
            self.field.get_boundary(
                r=0.95 * self.particle.r0,
                nphi=110,
                ntheta=30,
                ntheta_fourier=16,
                mpol=8,
                ntor=15,
            )
        )
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot3D(rpos_cartesian[0], rpos_cartesian[1], rpos_cartesian[2])
        ax.plot_surface(boundary[0], boundary[1], boundary[2], alpha=0.5)
        set_axes_equal(ax)
        ax.set_axis_off()
        ax.dist = distance
        plt.tight_layout()
        plt.show()


class particle_ensemble_orbit:
    r"""
    Interface function with the C++ executable NEAT. Receives a pyQSC instance
    and outputs the characteristics of the orbit.
    Args:
        stel: Qsc instance of pyQSC
        params (dict): a Python dict() containing the following parameters:
            r0,theta0,phi0,charge,rhom,mass,Lambda,energy,nsamples,Tfinal
        B20real (bool): True if a constant B20real should be used, False otherwise
    """

    def __init__(self, particles, field, nsamples=500, Tfinal=600, nthreads=8) -> None:

        self.particles = particles
        self.field = field
        self.nsamples = nsamples
        self.Tfinal = Tfinal
        self.nthreads = nthreads

        solution = np.array(
            gc_solver_qs_ensemble(
                *field.gyronimo_parameters(),
                *particles.gyronimo_parameters(),
                nsamples,
                Tfinal,
                nthreads
            )
        )
        self.gyronimo_parameters = solution
        self.time = solution[:, 0]
        self.nparticles = solution.shape[1] - 1
        self.r_pos = solution[:, 1:].transpose()

    def loss_fraction(self, r_surface_max=0.15):
        self.lost_times_of_particles = [
            self.time[np.argmax(particle_pos > r_surface_max)]
            for particle_pos in self.r_pos
        ]
        loss_fraction_array = [0.0]
        self.total_particles_lost = 0
        for time in self.time[1:]:
            self.total_particles_lost += self.lost_times_of_particles.count(time)
            loss_fraction_array.append(self.total_particles_lost / self.nparticles)
        self.loss_fraction_array = loss_fraction_array
        return loss_fraction_array

    def plot_loss_fraction(self):
        try:
            self.loss_fraction_array
        except NameError:
            print(
                "First calculate the loss fraction using the self.loss_fraction() function"
            )
        else:
            plt.semilogx(self.time, self.loss_fraction_array)
            plt.xlabel("Time")
            plt.ylabel("Loss Fraction")
            plt.tight_layout()
            plt.show()


def canonical_angular_momentum(particle, field, r_pos, v_parallel, Bfield):

    m_proton = PROTON_MASS
    e = ELEMENTARY_CHARGE
    mu0 = MU_0
    Valfven = field.Bbar / np.sqrt(mu0 * particle.rhom * m_proton * 1.0e19)

    p_phi1 = (
        particle.mass
        * m_proton
        * v_parallel
        * Valfven
        * (field.G0 + r_pos**2 * (field.G2 + (field.iota - field.iotaN) * field.I2))
        / Bfield
        / field.Bbar
    )

    p_phi2 = particle.charge * e * r_pos**2 * field.Bbar / 2 * field.iotaN
    p_phi = p_phi1 - p_phi2

    return p_phi
