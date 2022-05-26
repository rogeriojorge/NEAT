import logging

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline as spline

from .constants import ELEMENTARY_CHARGE, MU_0, PROTON_MASS
from .fields import stellna_qs

logger = logging.getLogger(__name__)


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Args:
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


class charged_particle:
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
        r0=0.05,
        theta0=np.pi,
        phi0=0,
    ) -> None:
        self.charge = charge
        self.mass = mass
        self.energy = energy
        self.Lambda = Lambda
        self.vpp_sign = vpp_sign
        self.r0 = r0
        self.theta0 = theta0
        self.phi0 = phi0

    def gyronimo_parameters(self):
        return (
            self.charge,
            self.mass,
            self.Lambda,
            self.vpp_sign,
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
        charge=2,
        mass=4,
        energy=3.52e6,
        nlambda_trapped=10,
        nlambda_passing=3,
        r0=0.05,
        r_max=0.1,
        ntheta=10,
        nphi=10,
    ) -> None:
        self.charge = charge
        self.mass = mass
        self.energy = energy
        self.nlambda_trapped = nlambda_trapped
        self.nlambda_passing = nlambda_passing
        self.r0 = r0
        self.r_max = r_max
        self.ntheta = ntheta
        self.nphi = nphi

    def gyronimo_parameters(self):
        return (
            self.charge,
            self.mass,
            self.energy,
            self.nlambda_trapped,
            self.nlambda_passing,
            self.r0,
            self.r_max,
            self.ntheta,
            self.nphi,
        )


class particle_orbit:
    r"""
    Interface function with the C++ executable NEAT. Receives a pyQSC instance
    and outputs the characteristics of the orbit.
    Args:
        stel: Qsc instance of pyQSC
        params (dict): a Python dict() containing the following parameters:
            r0,theta0,phi0,charge,mass,Lambda,energy,nsamples,Tfinal
        B20real (bool): True if a constant B20real should be used, False otherwise
    """

    def __init__(self, particle, field, nsamples=1000, Tfinal=0.0001, B20_constant=False) -> None:

        self.particle = particle
        self.field = field
        self.nsamples = nsamples
        self.Tfinal = Tfinal

        self.field.B20_constant = B20_constant

        solution = np.array(
            self.field.neatpp_solver(
                *self.field.gyronimo_parameters(),
                *self.particle.gyronimo_parameters(),
                self.nsamples,
                self.Tfinal
            )
        )

        self.gyronimo_parameters = solution

        nu = field.varphi - field.phi
        nu_spline_of_varphi = spline(
            np.append(field.varphi, 2 * np.pi / field.nfp),
            np.append(nu, nu[0]),
            bc_type="periodic",
        )

        self.time = solution[:, 0]
        self.r_pos = solution[:, 1]
        self.theta_pos = solution[:, 2]
        self.varphi_pos = solution[:, 3]
        self.phi_pos = self.varphi_pos - nu_spline_of_varphi(self.varphi_pos)
        self.energy_parallel = solution[:, 4]
        self.energy_perpendicular = solution[:, 5]
        self.total_energy = self.energy_parallel + self.energy_perpendicular

        self.Bfield = solution[:, 6]
        self.v_parallel = solution[:, 7]
        self.rdot = solution[:, 8]
        self.thetadot = solution[:, 9]
        self.varphidot = solution[:, 10]
        self.vparalleldot = solution[:, 11]

        self.p_phi = canonical_angular_momentum(
            particle, field, self.r_pos, self.v_parallel, self.Bfield
        )

        self.rpos_cylindrical = np.array(
            self.field.to_RZ(
                np.array([self.r_pos, self.theta_pos, self.varphi_pos]).transpose()
            )
        )

        self.rpos_cartesian = np.array(
            [
                self.rpos_cylindrical[0] * np.cos(self.rpos_cylindrical[2]),
                self.rpos_cylindrical[0] * np.sin(self.rpos_cylindrical[2]),
                self.rpos_cylindrical[1],
            ]
        )

    def plot_orbit(self, show=True):
        x = self.r_pos * np.cos(self.theta_pos)
        y = self.r_pos * np.sin(self.theta_pos)
        plt.figure()
        plt.plot(x, y)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel(r"r cos($\theta$)")
        plt.ylabel(r"r sin($\theta$)")
        plt.tight_layout()
        if show:
            plt.show()

    def plot_orbit_3D(self, r_surface=0.1, distance=6, show=True):
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
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig = plt.figure(figsize=(10, 3))

        ax = fig.add_subplot(131, projection="3d")
        ax.plot3D(
            self.rpos_cartesian[0], self.rpos_cartesian[1], self.rpos_cartesian[2]
        )
        ax.plot_surface(boundary[0], boundary[1], boundary[2], alpha=0.5)
        set_axes_equal(ax)
        ax.set_axis_off()
        ax.dist = distance

        ax = fig.add_subplot(132, projection="3d")
        ax.plot3D(
            self.rpos_cartesian[0], self.rpos_cartesian[1], self.rpos_cartesian[2]
        )
        ax.plot_surface(boundary[0], boundary[1], boundary[2], alpha=0.5)
        set_axes_equal(ax)
        ax.set_axis_off()
        ax.view_init(azim=90, elev=90)
        ax.dist = distance

        ax = fig.add_subplot(133, projection="3d")
        ax.plot3D(
            self.rpos_cartesian[0], self.rpos_cartesian[1], self.rpos_cartesian[2]
        )
        ax.plot_surface(boundary[0], boundary[1], boundary[2], alpha=0.5)
        set_axes_equal(ax)
        ax.set_axis_off()
        ax.view_init(azim=0, elev=0)
        ax.dist = distance - 1

        plt.tight_layout()
        if show:
            plt.show()

    def plot(self, show=True):
        fig = plt.figure(figsize=(10, 6))
        plt.subplot(3, 3, 1)
        plt.plot(self.time, self.r_pos)
        plt.xlabel(r"$t (s)$")
        plt.ylabel(r"$r$")
        plt.subplot(3, 3, 2)
        plt.plot(self.time, self.theta_pos)
        plt.xlabel(r"$t (s)$")
        plt.ylabel(r"$\theta$")
        plt.subplot(3, 3, 3)
        plt.plot(self.time, self.varphi_pos)
        plt.xlabel(r"$t (s)$")
        plt.ylabel(r"$\varphi$")
        plt.subplot(3, 3, 4)
        plt.plot(self.time, self.v_parallel)
        plt.xlabel(r"$t (s)$")
        plt.ylabel(r"$v_\parallel$")
        plt.subplot(3, 3, 5)
        plt.plot(
            self.time, (self.total_energy - self.total_energy[0]) / self.total_energy[0]
        )
        plt.xlabel(r"$t (s)$")
        plt.ylabel(r"$(E-E_0)/E_0$")
        plt.subplot(3, 3, 6)
        plt.plot(self.time, (self.p_phi - self.p_phi[0]) / self.p_phi[0])
        plt.xlabel(r"$t (s)$")
        plt.ylabel(r"$(p_\phi-p_{\phi0})/p_{\phi0}$")
        plt.subplot(3, 3, 7)
        plt.plot(self.time, self.rdot, label=r"$\dot r$")
        plt.plot(self.time, self.thetadot, label=r"$\dot \theta$")
        plt.plot(self.time, self.varphidot, label=r"$\dot \varphi$")
        plt.plot(self.time, self.vparalleldot, label=r"$\dot v_\parallel$")
        plt.xlabel(r"$t (s)$")
        plt.legend()
        plt.subplot(3, 3, 8)
        plt.plot(
            self.r_pos * np.cos(self.theta_pos), self.r_pos * np.sin(self.theta_pos)
        )
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel(r"r cos($\theta$)")
        plt.ylabel(r"r sin($\theta$)")
        plt.subplot(3, 3, 9)
        plt.plot(self.rpos_cylindrical[0], self.rpos_cylindrical[1])
        plt.xlabel(r"$R$")
        plt.ylabel(r"$Z$")
        plt.tight_layout()
        if show:
            plt.show()

    def plot_animation(self, r_surface=0.1, distance=7, show=True, SaveMovie=False):
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection="3d")

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

        ax.plot_surface(
            boundary[0],
            boundary[1],
            boundary[2],
            rstride=1,
            cstride=1,
            antialiased=False,
            linewidth=0,
            alpha=0.15,
        )
        ax.auto_scale_xyz(
            [boundary[0].min(), boundary[0].max()],
            [boundary[0].min(), boundary[0].max()],
            [boundary[0].min(), boundary[0].max()],
        )
        ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.set_axis_off()
        ax.dist = distance

        ani = []

        def update(num, data, line):
            line.set_data(data[:2, 0:num])
            line.set_3d_properties(data[2, 0:num])

        (line,) = ax.plot(
            self.rpos_cartesian[0][0:1],
            self.rpos_cartesian[1][0:1],
            self.rpos_cartesian[2][0:1],
            lw=2,
        )
        ani = animation.FuncAnimation(
            fig,
            update,
            self.nsamples,
            fargs=(self.rpos_cartesian, line),
            interval=self.nsamples / 200,
        )

        if show:
            plt.show()

        if SaveMovie:
            ani.save(
                "particle_Orbit.mp4",
                fps=30,
                dpi=300,
                codec="libx264",
                bitrate=-1,
                extra_args=["-pix_fmt", "yuv420p"],
            )


class particle_ensemble_orbit:
    r"""
    Interface function with the C++ executable NEAT. Receives a pyQSC instance
    and outputs the characteristics of the orbit.
    Args:
        stel: Qsc instance of pyQSC
        params (dict): a Python dict() containing the following parameters:
            r0,theta0,phi0,charge,mass,Lambda,energy,nsamples,Tfinal
        B20real (bool): True if a constant B20real should be used, False otherwise
    """

    def __init__(
        self,
        particles: charged_particle_ensemble,
        field: stellna_qs,
        nsamples=800,
        Tfinal=0.0001,
        nthreads=2,
    ) -> None:

        self.particles = particles
        self.field = field
        self.nsamples = nsamples
        self.nthreads = nthreads
        self.Tfinal = Tfinal

        solution = np.array(
            self.field.neatpp_solver_ensemble(
                *self.field.gyronimo_parameters(),
                *self.particles.gyronimo_parameters(),
                self.nsamples,
                self.Tfinal,
                self.nthreads
            )
        )
        self.gyronimo_parameters = solution
        self.time = solution[:, 0]
        self.nparticles = solution.shape[1] - 1
        self.r_pos = solution[:, 1:].transpose()

        # Save the values of theta, phi and lambda used
        r_max = self.particles.r_max
        self.B_max = (
            abs(field.B0)
            + abs(r_max * field.B1c)
            + r_max * r_max * (abs(field.B20_mean) + abs(field.B2c))
        )
        self.B_min = max(
            0.01,
            abs(field.B0)
            - abs(r_max * field.B1c)
            - r_max * r_max * (abs(field.B20_mean) + abs(field.B2c)),
        )
        self.theta = np.linspace(0.0, 2 * np.pi, particles.ntheta)
        self.phi = np.linspace(0.0, 2 * np.pi / field.nfp, particles.nphi)
        self.lambda_trapped = np.linspace(
            field.B0 / self.B_max, field.B0 / self.B_min, particles.nlambda_trapped
        )
        self.lambda_passing = np.linspace(
            0.0,
            field.B0 / self.B_max * (1.0 - 1.0 / particles.nlambda_passing),
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
            r = particles.r0
            magB = (
                field.B0
                + r * (field.B1c * np.cos(theta))
                + r * r * (field.B20_mean + field.B2c * np.cos(2 * theta))
            )
            self.initial_jacobian.append(
                (field.G0 + r * r * field.G2 + field.iota * field.I2)
                * r
                * field.B0
                / magB
            )

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

    def plot_loss_fraction(self, show=True):
        plt.semilogx(self.time, self.loss_fraction_array)
        plt.xlabel("Time (s)")
        plt.ylabel("Loss Fraction")
        plt.tight_layout()
        if show:
            plt.show()


def canonical_angular_momentum(particle, field, r_pos, v_parallel, Bfield):

    m_proton = PROTON_MASS
    e = ELEMENTARY_CHARGE

    p_phi1 = (
        particle.mass
        * m_proton
        * v_parallel
        * (field.G0 + r_pos**2 * (field.G2 + (field.iota - field.iotaN) * field.I2))
        / Bfield
        / field.Bbar
    )

    p_phi2 = particle.charge * e * r_pos**2 * field.Bbar / 2 * field.iotaN
    p_phi = p_phi1 - p_phi2

    return p_phi
