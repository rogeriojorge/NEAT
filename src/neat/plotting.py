""" Plotting module of NEAT

This script defines the necessary plotting
functions to show particle orbits and their
attributes for NEAT.

"""

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import netcdf


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
    ax.plot_surface(boundary[0], boundary[1], boundary[2], alpha=0.25)
    set_axes_equal(ax)
    ax.set_axis_off()
    ax.dist = distance

    ax = fig.add_subplot(132, projection="3d")
    ax.plot3D(rpos_cartesian[0], rpos_cartesian[1], rpos_cartesian[2])
    ax.plot_surface(boundary[0], boundary[1], boundary[2], alpha=0.25)
    set_axes_equal(ax)
    ax.set_axis_off()
    ax.view_init(azim=90, elev=90)
    ax.dist = distance

    ax = fig.add_subplot(133, projection="3d")
    ax.plot3D(rpos_cartesian[0], rpos_cartesian[1], rpos_cartesian[2])
    ax.plot_surface(boundary[0], boundary[1], boundary[2], alpha=0.25)
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
    plt.figure(figsize=(10, 6))
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
    # plt.plot(self.time, self.vparalleldot, label=r"$\dot v_\parallel$")
    plt.xlabel(r"$t (s)$")
    plt.legend()
    plt.subplot(3, 3, 8)
    plt.plot(self.r_pos * np.cos(self.theta_pos), self.r_pos * np.sin(self.theta_pos))
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel(r"r cos($\theta$)")
    plt.ylabel(r"r sin($\theta$)")
    plt.subplot(3, 3, 9)
    # plt.plot(self.rpos_cylindrical[0], self.rpos_cylindrical[1])
    # plt.xlabel(r"$R$")
    # plt.ylabel(r"$Z$")
    plt.plot(self.time, self.magnetic_field_strength)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$|B|$")
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


def get_vmec_boundary(wout_filename):  # pylint: disable=R0914
    """Obtain (X, Y, Z) of a magnetic flux surface from a vmec equilibrium"""
    net_file = netcdf.netcdf_file(wout_filename, "r", mmap=False)
    nsurfaces = net_file.variables["ns"][()]
    nfp = net_file.variables["nfp"][()]
    xn = net_file.variables["xn"][()]  # pylint: disable=C0103
    xm = net_file.variables["xm"][()]  # pylint: disable=C0103
    xn_nyq = net_file.variables["xn_nyq"][()]
    xm_nyq = net_file.variables["xm_nyq"][()]
    rmnc = net_file.variables["rmnc"][()]
    zmns = net_file.variables["zmns"][()]
    bmnc = net_file.variables["bmnc"][()]
    lasym = net_file.variables["lasym__logical__"][()]
    if lasym == 1:
        rmns = net_file.variables["rmns"][()]
        zmnc = net_file.variables["zmnc"][()]
        bmns = net_file.variables["bmns"][()]
    else:
        rmns = 0 * rmnc
        zmnc = 0 * rmnc
        bmns = 0 * bmnc
    net_file.close()
    nmodes = len(xn)

    ntheta = 50
    nzeta = int(90 * nfp)
    zeta_2d, theta_2d = np.meshgrid(
        np.linspace(0, 2 * np.pi, num=nzeta), np.linspace(0, 2 * np.pi, num=ntheta)
    )
    iradius = nsurfaces - 1
    r_coordinate = np.zeros((ntheta, nzeta))
    z_coordinate = np.zeros((ntheta, nzeta))
    b_field = np.zeros((ntheta, nzeta))
    for imode in range(nmodes):
        angle = xm[imode] * theta_2d - xn[imode] * zeta_2d
        r_coordinate = (
            r_coordinate
            + rmnc[iradius, imode] * np.cos(angle)
            + rmns[iradius, imode] * np.sin(angle)
        )
        z_coordinate = (
            z_coordinate
            + zmns[iradius, imode] * np.sin(angle)
            + zmnc[iradius, imode] * np.cos(angle)
        )

    for imode, xn_nyq_i in enumerate(xn_nyq):
        angle = xm_nyq[imode] * theta_2d - xn_nyq_i * zeta_2d
        b_field = (
            b_field
            + bmnc[iradius, imode] * np.cos(angle)
            + bmns[iradius, imode] * np.sin(angle)
        )

    x_coordinate = r_coordinate * np.cos(zeta_2d)
    y_coordinate = r_coordinate * np.sin(zeta_2d)

    b_rescaled = (b_field - b_field.min()) / (b_field.max() - b_field.min())

    return [x_coordinate, y_coordinate, z_coordinate], b_rescaled


def get_vmec_magB(
    wout_filename, spos=None, ntheta=50, nzeta=100
):  # pylint: disable=R0914  
    """Obtain contours of B on a magnetic flux surface from a vmec equilibrium"""
    net_file = netcdf.netcdf_file(wout_filename, "r", mmap=False)
    nsurfaces = net_file.variables["ns"][()]
    xn_nyq = net_file.variables["xn_nyq"][()]
    xm_nyq = net_file.variables["xm_nyq"][()]
    bmnc = net_file.variables["bmnc"][()]
    lasym = net_file.variables["lasym__logical__"][()]
    if lasym == 1:
        bmns = net_file.variables["bmns"][()]
    else:
        bmns = 0 * bmnc
    net_file.close()

    zeta_2d, theta_2d = np.meshgrid(
        np.linspace(0, 2 * np.pi, num=nzeta), np.linspace(0, 2 * np.pi, num=ntheta)
    )

    if not spos:
        iradius = nsurfaces - 1
    else:
        iradius = int(nsurfaces * spos)

    if (spos!=None and (spos<=0 or spos>=1)):
        print("Value spos must be higher than 0 and lower than 1")
        exit()

    b_field = np.zeros((ntheta, nzeta))

    for imode, xn_nyq_i in enumerate(xn_nyq):
        angle = xm_nyq[imode] * theta_2d - xn_nyq_i * zeta_2d
        b_field = (
            b_field
            + bmnc[iradius, imode] * np.cos(angle)
            + bmns[iradius, imode] * np.sin(angle)
        )

    return b_field
