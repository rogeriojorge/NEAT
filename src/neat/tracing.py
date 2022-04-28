import logging
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline as spline

from neatpp import gc_solver_qs, gc_solver_qs_ensemble

from .constants import ELEMENTARY_CHARGE, MU_0, PROTON_MASS

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

    def plot_orbit_3D(self, distance=6, show=True):
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
        plt.xlabel(r"$t$")
        plt.ylabel(r"$r$")
        plt.subplot(3, 3, 2)
        plt.plot(self.time, self.theta_pos)
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\theta$")
        plt.subplot(3, 3, 3)
        plt.plot(self.time, self.varphi_pos)
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\varphi$")
        plt.subplot(3, 3, 4)
        plt.plot(self.time, self.v_parallel)
        plt.xlabel(r"$t$")
        plt.ylabel(r"$v_\parallel$")
        plt.subplot(3, 3, 5)
        plt.plot(
            self.time, (self.total_energy - self.total_energy[0]) / self.total_energy[0]
        )
        plt.xlabel(r"$t$")
        plt.ylabel(r"$(E-E_0)/E_0$")
        plt.subplot(3, 3, 6)
        plt.plot(self.time, (self.p_phi - self.p_phi[0]) / self.p_phi[0])
        plt.xlabel(r"$t$")
        plt.ylabel(r"$(p_\phi-p_{\phi0})/p_{\phi0}$")
        plt.subplot(3, 3, 7)
        plt.plot(self.time, self.rdot, label=r"$\dot r$")
        plt.plot(self.time, self.thetadot, label=r"$\dot \theta$")
        plt.plot(self.time, self.varphidot, label=r"$\dot \varphi$")
        plt.plot(self.time, self.vparalleldot, label=r"$\dot v_\parallel$")
        plt.xlabel(r"$t$")
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

    def plot_animation(self, show=True, SaveMovie=False):
        fig = plt.figure(frameon=False, figsize=(10, 5))
        # fig.set_size_inches(10, 5)
        ax = p3.Axes3D(fig)

        start_time = time.time()
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
        ax.dist = 6.5

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
            interval=self.Tfinal / self.nsamples,
        )

        if show:
            plt.show()

        if SaveMovie:
            start_time = time.time()
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

    def plot_loss_fraction(self, show=True):
        plt.semilogx(self.time, self.loss_fraction_array)
        plt.xlabel("Time")
        plt.ylabel("Loss Fraction")
        plt.tight_layout()
        if show:
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