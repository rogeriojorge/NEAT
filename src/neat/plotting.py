""" Plotting module of NEAT

This script defines the necessary plotting
functions to show particle orbits and their
attributes for NEAT.

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

## Uncomment the two lines below if the 3D
## plotting/animation is not working for some reason
# import mpl_toolkits.mplot3d.axes3d as p3
# from mpl_toolkits.mplot3d import Axes3D


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


def plot_orbit2d(x_position, y_position, show=True):
    """
    Make a plot of a single particle orbit in
    (r,theta) coordinates where r is the square
    root of the toroidal magnetic flux and theta
    the poloidal Boozer angle.
    """
    plt.figure()
    plt.plot(x_position, y_position)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel(r"r cos($\theta$)")
    plt.ylabel(r"r sin($\theta$)")
    plt.tight_layout()
    if show:
        plt.show()


def plot_orbit3d(boundary, rpos_cartesian, distance=6, show=True):
    """
    Make a three-dimensional plot of a single particle orbit
    together with the corresponding stellarator toroidal flux
    surface given by boundary.
    """
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(131, projection="3d")

    ax.plot3D(rpos_cartesian[0], rpos_cartesian[1], rpos_cartesian[2])

    ax.plot_surface(boundary[0], boundary[1], boundary[2], alpha=0.5)
    set_axes_equal(ax)
    ax.set_axis_off()
    ax.dist = distance

    ax = fig.add_subplot(132, projection="3d")
    ax.plot3D(rpos_cartesian[0], rpos_cartesian[1], rpos_cartesian[2])
    ax.plot_surface(boundary[0], boundary[1], boundary[2], alpha=0.5)
    set_axes_equal(ax)
    ax.set_axis_off()
    ax.view_init(azim=90, elev=90)
    ax.dist = distance

    ax = fig.add_subplot(133, projection="3d")
    ax.plot3D(rpos_cartesian[0], rpos_cartesian[1], rpos_cartesian[2])
    ax.plot_surface(boundary[0], boundary[1], boundary[2], alpha=0.5)
    set_axes_equal(ax)
    ax.set_axis_off()
    ax.view_init(azim=0, elev=0)
    ax.dist = distance - 1

    plt.tight_layout()
    if show:
        plt.show()


def plot_parameters(self, show=True):
    """
    Make a single plot with relevant physics parameters
    of a single particle orbit on a magnetic field.
    """
    _ = plt.figure(figsize=(10, 6))
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
    plt.ylabel(r"$(p_\phi-p_{\phi_initial})/p_{\phi_initial}$")
    plt.subplot(3, 3, 7)
    plt.plot(self.time, self.rdot, label=r"$\dot r$")
    plt.plot(self.time, self.thetadot, label=r"$\dot \theta$")
    plt.plot(self.time, self.varphidot, label=r"$\dot \varphi$")
    plt.plot(self.time, self.vparalleldot, label=r"$\dot v_\parallel$")
    plt.xlabel(r"$t (s)$")
    plt.legend()
    plt.subplot(3, 3, 8)
    plt.plot(self.r_pos * np.cos(self.theta_pos), self.r_pos * np.sin(self.theta_pos))
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


def plot_animation3d(
    boundary, rpos_cartesian, nsamples, distance=7, show=True, save_movie=False
):
    """
    Show a three-dimensional animation of a particle
    orbit together with a flux surface of the stellarator
    """
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")
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
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)  # pylint: disable=W0212
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)  # pylint: disable=W0212
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)  # pylint: disable=W0212
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
        rpos_cartesian[0][0:1],
        rpos_cartesian[1][0:1],
        rpos_cartesian[2][0:1],
        lw=2,
    )
    ani = animation.FuncAnimation(
        fig,
        update,
        nsamples,
        fargs=(rpos_cartesian, line),
        interval=nsamples / 200,
    )

    if show:
        plt.show()

    if save_movie:
        ani.save(
            "ParticleOrbit.mp4",
            fps=30,
            dpi=300,
            codec="libx264",
            bitrate=-1,
            extra_args=["-pix_fmt", "yuv420p"],
        )
