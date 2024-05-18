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
from scipy.io import netcdf_file

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


def plot_orbit2d(x_position, y_position, show=True, savefig=None):
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
    
    if savefig is not None:
        plt.savefig(savefig)
    if show:
        plt.show()

def plot_orbit3d(boundary, rpos_cartesian, distance=6, show=True, savefig=None):
    """
    Make a three-dimensional plot of a single particle orbit
    together with the corresponding stellarator toroidal flux
    surface given by boundary.
    """
    fig = plt.figure(figsize=(20, 6))
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
    
    if savefig is not None:
        plt.savefig(savefig)
    if show:
        plt.show()
    plt.close()


def plot_parameters(self, r_minor=1.0, show=True, savefig=None):
    """
    Make a single plot with relevant physics parameters
    of a single particle orbit on a magnetic field.
    """
    from scipy import signal
    from matplotlib import patches

    # if r_minor!=1:
    #     norm_r_pos=(self.r_pos/r_minor)**2
    # else: norm_r_pos=self.r_pos

    if self.field.near_axis:
        norm_r_pos=(self.r_pos/r_minor)**2
        self_theta_pos = -( 
            np.pi + self.theta_pos + (self.field.iota - self.field.iotaN) * self.varphi_pos
        )
    else:
        self_theta_pos = self.theta_pos
        norm_r_pos=self.r_pos

    v_valleys, _ = signal.find_peaks(
        -np.abs(self.v_parallel), distance=(1 / 100) * self.time.size
    )
    v_valleys_0 = v_valleys[(np.abs(self.v_parallel[v_valleys]) < 1e5)]

    phases = self.varphi_pos
    phases = (phases + np.pi) % (2 * np.pi) - np.pi

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams["figure.facecolor"] = "w"
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['lines.markersize'] = 15
    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)
    plt.rc('font', size=24)
    plt.rc('legend', fontsize=18)
    plt.rc('lines', linewidth=3)

    plt.figure(figsize=(16, 16))
    plt.subplot(3, 3, 1)
    plt.plot(
        self.time*1e6, 
        norm_r_pos
        )
    plt.xlabel(r"$t \ (\mu s)$")
    plt.ylabel(r"$s=\psi/\psi_b$")
    plt.subplot(3, 3, 2)
    plt.plot(
        self.rpos_cylindrical[0]*np.cos(self.rpos_cylindrical[2]),
        self.rpos_cylindrical[0]*np.sin(self.rpos_cylindrical[2])
        )
    plt.xlabel(r'$R \ \cos{\Phi}$ (m)')
    plt.ylabel(r'$R \ \sin{\Phi}$ (m)')
    ax1=plt.subplot(3, 3, 3)
    plt.plot(
        norm_r_pos*np.cos(self_theta_pos), 
        norm_r_pos*np.sin(self_theta_pos)
        )
    circle=patches.Circle((0,0), radius=1,color='black',fill=False,linestyle='dotted',linewidth=1)
    circle2=patches.Circle((0,0), radius=norm_r_pos[0],color='black',fill=False,linestyle='dotted',linewidth=1)
    ax1.add_patch(circle)
    ax1.add_patch(circle2)
    ax1.set(xlim=(-1.2,1.2),ylim=(-1.2,1.2))
    plt.xlabel(r'$s \ \cos{\theta}$')
    plt.ylabel(r'$s \ \sin{\theta}$')
    plt.subplot(3, 3, 4)
    plt.plot(self.time * 1e6, self.v_parallel)
    plt.plot(
        self.time[v_valleys] * 1e6,
        self.v_parallel[v_valleys],
        color="blue",
        marker=".",
        linestyle="None",
    )
    plt.plot(
        self.time[v_valleys_0] * 1e6,
        self.v_parallel[v_valleys_0],
        color="red",
        marker=".",
        linestyle="None",
    )
    plt.xlabel(r"$t \ (\mu s)$")
    plt.ylabel(r"$v_\parallel$ (m/s)")
    plt.subplot(3, 3, 5)
    plt.plot(
        self.time * 1e6,
        (self.total_energy - self.total_energy[0]) / self.total_energy[0],
    )
    plt.xlabel(r"$t \ (\mu s)$")
    plt.ylabel(r"$(E-E_0)/E_0$")
    plt.subplot(3, 3, 6)
    plt.plot(
        self.time * 1e6,
        (self.p_phi - self.p_phi[0]) / self.p_phi[0]
        )
    plt.xlabel(r"$t \ (\mu s)$")
    plt.ylabel(r"$(p_\phi-p_{\phi_0})/p_{\phi_0}$")
    plt.subplot(3, 3, 7)
    plt.plot(
        self.time * 1e6,
        self.rdot*(10**(-3)), 
        label=r"$\dot r$"
        )
    plt.plot(
        self.time * 1e6,
        self.thetadot*(10**(-3)), 
        label=r"$\dot \theta$"
        )
    plt.plot(
        self.time * 1e6,
        self.varphidot*(10**(-3)), 
        label=r"$\dot \varphi$"
        )
    # plt.plot(self.time * 1e6, self.vparalleldot, label=r"$\dot v_\parallel$")
    plt.xlabel(r"$t$")
    plt.ylabel(r"Coordinates Velocity ($\times 10^3$)")
    plt.xlabel(r"$t \ (\mu s)$")
    plt.legend()
    plt.subplot(3, 3, 8)
    plt.plot(
        self.time * 1e6, 
        self.magnetic_field_strength
        )
    plt.xlabel(r"$t \ (\mu s)$")
    plt.ylabel(r"$|B|$ (T)")
    plt.subplot(3, 3, 9)
    if r_minor != 1:
        norm_r_pos_v = (self.r_pos[v_valleys_0] / r_minor) ** 2
        plt.plot(
            phases[v_valleys_0], 
            norm_r_pos_v, 
            c="red", 
            marker=".", 
            ls="None", 
            label='Turning points'
        )
    else:
        plt.plot(
            phases[v_valleys_0],
            self.r_pos[v_valleys_0],
            color="red",
            marker=".",
            linestyle="None", 
            label='Turning points',
        )
    plt.xlabel(r"$\phi$ (rad)")
    plt.ylabel(r"$s$")
    plt.ylim(0, 1)
    plt.xlim(-np.pi, np.pi)
    plt.legend(handlelength=1.3, labelspacing=0.7, columnspacing=1.4, loc=2)
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    if show:
        plt.show()
    plt.close()


def update(num, data, line):
    line.set_data(data[:2, 0:num])
    line.set_3d_properties(data[2, 0:num])


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
    net_file = netcdf_file(wout_filename, "r", mmap=False)
    try:
        nsurfaces = net_file.variables["ns"][()]
        nfp = net_file.variables["nfp"][()]
        xn = net_file.variables["xn"][()]  # pylint: disable=C0103
        xm = net_file.variables["xm"][()]  # pylint: disable=C0103
        xn_nyq = net_file.variables["xn_nyq"][()]
        xm_nyq = net_file.variables["xm_nyq"][()]
        rmnc = net_file.variables["rmnc"][()]
        zmns = net_file.variables["zmns"][()]
        bmnc = net_file.variables["bmnc"][()]
    except:
        nsurfaces = net_file.variables["ns_b"][()]
        nsurfaces -= 1
        nfp = net_file.variables["nfp_b"][()]
        xn = net_file.variables["ixn_b"][()]  # pylint: disable=C0103
        xm = net_file.variables["ixm_b"][()]  # pylint: disable=C0103
        xn_nyq = xn
        xm_nyq = xm
        rmnc = net_file.variables["rmnc_b"][()]
        zmns = net_file.variables["zmns_b"][()]
        bmnc = net_file.variables["bmnc_b"][()]
        print(len(bmnc))
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
    net_file = netcdf_file(wout_filename, "r", mmap=False)
    try:
        nsurfaces = net_file.variables["ns"][()]
        xn_nyq = net_file.variables["xn_nyq"][()]
        xm_nyq = net_file.variables["xm_nyq"][()]
        bmnc = net_file.variables["bmnc"][()]
        lasym = net_file.variables["lasym__logical__"][()]
    except:
        nsurfaces = net_file.variables["ns_b"][()]
        xn_nyq = net_file.variables["ixn_b"][()]
        xm_nyq = net_file.variables["ixm_b"][()]
        bmnc = net_file.variables["bmnc_b"][()]
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

    if spos != None and (spos <= 0 or spos >= 1):
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


def butter_lowpass_filter2(
    data, cutoff, fs, half_order=5, axis=-1, handle_nans=True, keep_nans=True
):
    from scipy import signal
    from scipy.interpolate import interp1d

    """This function applies a linear digital filter twice, once forward and once backwards. 
    The combined filter has zero phase and a filter order twice that of *half_order*.

    Parameters
    ----------
    data: ndarray
        Array of data to filter
    cutoff: float
        Cutoff frequency in Hz.
    fs: float
        Sampling frequency in Hz.
    half_order: int
        Half of the filter order.
    axis: int, optional
        The axis of x to which the filter is applied. Default is -1.
    handle_nans: bool, optional
        If True, NaN values are inter- and/or extrapolated before filtering. Defaults to False.
    keep_nans: bool, optional
        If True, values in the output are returned to NaN in the same locations as *data*. 
        
    Returns
    -------
    y: ndarray
        Filtered array

    Note: If *data* contains NaNs and *handle_nans*=False, the returned array will be all-NaNs.
    """

    y = np.array(data)

    if handle_nans:
        nan_mask = np.isnan(data)
        if nan_mask.any():
            if (~nan_mask).any():
                time = np.arange(len(data))
                y[nan_mask] = interp1d(
                    time[~nan_mask],
                    data[~nan_mask],
                    assume_sorted=True,
                    bounds_error=False,
                    fill_value="extrapolate",
                )(time[nan_mask])
            else:
                y[nan_mask] = np.nan

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = signal.butter(half_order, normal_cutoff, btype="low", output="sos")
    y = signal.sosfiltfilt(sos, y, axis)

    if handle_nans and keep_nans:
        y[nan_mask] = np.nan

    return y


def butter_lowpass_filter(data, cutoff, fs, half_order=5, axis=-1):
    from scipy import signal

    """This function applies a linear digital filter twice, once forward and once backwards. 
    The combined filter has zero phase and a filter order twice that of *half_order*.
    
    Parameters
    ----------
    data: ndarray
        Array of data to filter
    cutoff: float
        Cutoff frequency in Hz.
    fs: float
        Sampling frequency in Hz.
    half_order: int
        Half of the filter order.
    axis: int, optional
        The axis of x to which the filter is applied. Default is -1.
        
    Returns
    -------
    y: ndarray
        Filtered array
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = signal.butter(half_order, normal_cutoff, btype="low", output="sos")
    y = signal.sosfiltfilt(sos, data, axis)
    return y


def plot_diff_boozer(self, self2, r_minor, show=True, savefig=None):
    from matplotlib import patches
    from scipy import signal

    """
    Make a single plot with particle orbits on a magnetic field and their
    differences in cylindrical coordinates.
    """

    if r_minor != 1:
        norm_r_pos = (self.r_pos / r_minor) ** 2

    else:
        norm_r_pos = self.r_pos

    peaks, _ = signal.find_peaks(norm_r_pos, distance=(5 / 100) * self.time.size)
    valleys, _ = signal.find_peaks(-norm_r_pos, distance=(5 / 100) * self.time.size)

    if peaks.size > valleys.size:
        peaks = peaks[: valleys.size]
    else:
        valleys = valleys[: peaks.size]

    peaks2, _ = signal.find_peaks(self2.r_pos, distance=(7 / 100) * self.time.size)
    valleys2, _ = signal.find_peaks(-self2.r_pos, distance=(7 / 100) * self.time.size)

    if peaks2.size > valleys2.size:
        peaks2 = peaks2[: valleys2.size]
    else:
        valleys2 = valleys2[: peaks2.size]

    fs = 1 / (self.time[1] - self.time[0])  # Sampling frequency
    cutoff = 1e3  # Frequency cutoff value in Hz

    norm_r_pos_filt = butter_lowpass_filter(norm_r_pos, cutoff, fs)
    r_pos_filt2 = butter_lowpass_filter2(self2.r_pos, cutoff, fs)

    diff_s = norm_r_pos - self2.r_pos  # This needs to be changed
    diff_s_filt = norm_r_pos_filt - r_pos_filt2

    # Transforming theta_near-axis in theta_Boozer
    if self.field.near_axis:
        self_theta_pos = -( 
            np.pi + self.theta_pos + (self.field.iota - self.field.iotaN) * self.varphi_pos
        )
        label='NA'
    else:
        self_theta_pos = self.theta_pos
        label='Non-NA'

    diff_theta = (
        np.unwrap(np.mod(self_theta_pos, 2 * np.pi))
        - np.unwrap(np.mod(self2.theta_pos, 2 * np.pi))
    ) / (2 * np.pi)
    diff_Phi = (
        np.unwrap(np.mod(self.varphi_pos, 2 * np.pi))
        - np.unwrap(np.mod(self2.varphi_pos, 2 * np.pi))
    ) / (2 * np.pi)

    plt.figure(figsize=(20, 16))
    plt.subplot(3, 4, 1)
    plt.plot(self.time * 1e6, norm_r_pos)
    plt.plot(self.time * 1e6, norm_r_pos_filt)
    plt.plot(
        self.time[valleys] * 1e6,
        norm_r_pos[valleys],
        color="red",
        marker=".",
        linestyle="None",
    )
    plt.plot(
        self.time[peaks] * 1e6,
        norm_r_pos[peaks],
        color="red",
        marker=".",
        linestyle="None",
    )
    plt.xlabel(r"t ($\mu$s)")
    plt.ylabel(r"$s_1=\psi / \psi_b$")
    plt.subplot(3, 4, 2)
    plt.plot(self.time * 1e6, self_theta_pos)
    plt.xlabel(r"t ($\mu$s)")
    plt.ylabel(r"$\theta_1$ (rad)")
    plt.subplot(3, 4, 3)
    plt.plot(self.time * 1e6, self.varphi_pos)
    plt.xlabel(r"t ($\mu$s)")
    plt.ylabel(r"$\phi_1$ (rad)")
    ax = plt.subplot(3, 4, 4)
    plt.plot(norm_r_pos * np.cos(self_theta_pos), norm_r_pos * np.sin(self_theta_pos))
    circle = patches.Circle(
        (0, 0), radius=1, color="black", fill=False, linestyle="dotted", linewidth=1
    )
    circle2 = patches.Circle(
        (0, 0),
        radius=norm_r_pos[0],
        color="black",
        fill=False,
        linestyle="dotted",
        linewidth=1,
    )
    ax.add_patch(circle)
    ax.add_patch(circle2)
    ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
    plt.xlabel(r"$s_1 \ \cos{\theta_1}$")
    plt.ylabel(r"$s_1 \ \sin{\theta_1}$")
    plt.subplot(3, 4, 5)
    plt.plot(self2.time * 1e6, self2.r_pos)
    plt.plot(self2.time * 1e6, r_pos_filt2)
    plt.plot(
        self2.time[valleys2] * 1e6,
        self2.r_pos[valleys2],
        color="red",
        marker=".",
        linestyle="None",
    )
    plt.plot(
        self2.time[peaks2] * 1e6,
        self2.r_pos[peaks2],
        color="red",
        marker=".",
        linestyle="None",
    )
    plt.xlabel(r"t ($\mu$s)")
    plt.ylabel(r"$s_2=\psi / \psi_b$")
    plt.subplot(3, 4, 6)
    plt.plot(self2.time * 1e6, self2.theta_pos)
    plt.xlabel(r"t ($\mu$s)")
    plt.ylabel(r"$\theta_2$ (rad)")
    plt.subplot(3, 4, 7)
    plt.plot(self2.time * 1e6, self2.varphi_pos)
    plt.xlabel(r"t ($\mu$s)")
    plt.ylabel(r"$\phi_2$ (rad)")
    ax1 = plt.subplot(3, 4, 8)
    plt.plot(
        self2.r_pos * np.cos(self2.theta_pos), 
        self2.r_pos * np.sin(self2.theta_pos)
    )
    circle = patches.Circle(
        (0, 0), radius=1, color="black", fill=False, linestyle="dotted", linewidth=1
    )
    circle2 = patches.Circle(
        (0, 0),
        radius=self2.r_pos[0],
        color="black",
        fill=False,
        linestyle="dotted",
        linewidth=1,
    )
    ax1.add_patch(circle)
    ax1.add_patch(circle2)
    ax1.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
    plt.xlabel(r"$s_2 \ \cos{\theta_2}$")
    plt.ylabel(r"$s_2 \ \sin{\theta_2}$")
    plt.subplot(3, 4, 9)
    plt.plot(self2.time * 1e6, np.abs(diff_s))
    plt.plot(self2.time * 1e6, np.abs(diff_s_filt), label="Smoothed")
    plt.legend()
    plt.xlabel(r"t ($\mu$s)")
    plt.ylabel(r"$\Delta  s = s_1 - s_2$")
    plt.subplot(3, 4, 10)
    plt.plot(self2.time * 1e6, np.abs(diff_theta))
    plt.xlabel(r"t  ($\mu$s)")
    plt.ylabel(r"$\Delta \theta \ / \ (2 \pi)$")
    plt.subplot(3, 4, 11)
    plt.plot(self2.time * 1e6, np.abs(diff_Phi))
    plt.xlabel(r"t  ($\mu$s)")
    plt.ylabel(r"$\Delta \phi \ / \ (2 \pi)$")
    plt.subplot(3, 4, 12)
    avg_time = (self.time[peaks] + self.time[valleys]) * 1e6 / 2
    avg_time2 = (self2.time[peaks2] + self2.time[valleys2]) * 1e6 / 2
    plt.plot(avg_time, norm_r_pos[peaks] - norm_r_pos[valleys], label='Orbit 1', c='k')
    plt.plot(avg_time2, self2.r_pos[peaks2] - self2.r_pos[valleys2], label='Orbit 2', c='r')
    try:
        max = np.minimum(avg_time[-1], avg_time2[-1])
        plt.xlim(0, max)
    except:
        print("No radial oscillation")

    plt.xlabel(r"t ($\mu$s)")
    plt.ylabel(r"Orbit Width $\Delta s_i$")
    plt.legend()

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    if show:
        plt.show()
    plt.close()


def plot_diff_cyl(self, self2, show=True, savefig=None):
    """
    Make a single plot with particle orbits on a magnetic field and their
    differences in Boozer coordinates.
    """
    from scipy.interpolate import interp1d

    diff_r = (
        self.rpos_cylindrical[0][: self2.rpos_cylindrical[0].shape[0]]
        - self2.rpos_cylindrical[0][: self.rpos_cylindrical[0].shape[0]]
    ) / self2.rpos_cylindrical[0][: self.rpos_cylindrical[0].shape[0]]
    max_Z_vmec = np.nanmax(
        np.abs(self2.rpos_cylindrical[1][: self.rpos_cylindrical[0].shape[0]])
    )
    diff_Z = (
        self.rpos_cylindrical[1][: self2.rpos_cylindrical[0].shape[0]]
        - self2.rpos_cylindrical[1][: self.rpos_cylindrical[0].shape[0]]
    ) / max_Z_vmec
    diff_phi = (
        np.unwrap(
            np.mod(
                self.rpos_cylindrical[2][: self2.rpos_cylindrical[0].shape[0]],
                2 * np.pi,
            )
        )
        - np.unwrap(
            np.mod(
                self2.rpos_cylindrical[2][: self.rpos_cylindrical[0].shape[0]],
                2 * np.pi,
            )
        )
    ) / (2 * np.pi)

    _ = plt.figure(figsize=(20, 16))
    plt.subplot(3, 4, 1)
    plt.plot(self.time * 1e6, self.rpos_cylindrical[0])
    plt.xlabel(r"t ($\mu$s)")
    plt.ylabel(r"$R_1$ (m)")
    plt.subplot(3, 4, 2)
    plt.plot(self.time * 1e6, self.rpos_cylindrical[1])
    plt.xlabel(r"t  ($\mu$s)")
    plt.ylabel(r"$Z_1$ (m)")
    plt.subplot(3, 4, 3)
    plt.plot(self.time * 1e6, np.mod(self.rpos_cylindrical[2], 2 * np.pi))
    plt.xlabel(r"t  ($\mu$s)")
    plt.ylabel(r"$\Phi_1$ (rad)")
    plt.subplot(3, 4, 4)
    plt.plot(
        self.rpos_cylindrical[0] * np.cos(self.rpos_cylindrical[2]),
        self.rpos_cylindrical[0] * np.sin(self.rpos_cylindrical[2]),
    )
    plt.xlabel(r"$R_1 \ \cos{\Phi_1}$ (m)")
    plt.ylabel(r"$R_1 \ \sin{\Phi_1}$ (m)")
    plt.subplot(3, 4, 5)
    plt.plot(self2.time * 1e6, self2.rpos_cylindrical[0])
    plt.xlabel(r"t ($\mu$s)")
    plt.ylabel(r"$R_2$ (m)")
    plt.subplot(3, 4, 6)
    plt.plot(self2.time * 1e6, self2.rpos_cylindrical[1])
    plt.xlabel(r"t  ($\mu$s)")
    plt.ylabel(r"$Z_2$ (m)")
    plt.subplot(3, 4, 7)
    plt.plot(self2.time * 1e6, np.mod(self2.rpos_cylindrical[2], 2 * np.pi))
    plt.xlabel(r"t  ($\mu$s)")
    plt.ylabel(r"$\Phi_2$ (rad)")
    plt.subplot(3, 4, 8)
    plt.plot(
        self2.rpos_cylindrical[0] * np.cos(self2.rpos_cylindrical[2]),
        self2.rpos_cylindrical[0] * np.sin(self2.rpos_cylindrical[2]),
    )
    plt.xlabel(r"$R_2 \ \cos{\Phi_2}$ (m)")
    plt.ylabel(r"$R_2 \ \sin{\Phi_2}$ (m)")
    plt.subplot(3, 4, 9)
    plt.plot(self2.time * 1e6, np.abs(diff_r) * 100)
    plt.xlabel(r"t ($\mu$s)")
    plt.ylabel(r"$\Delta  R \ / \ R_{max} \ (\%)$")
    plt.subplot(3, 4, 10)
    plt.plot(self2.time * 1e6, np.abs(diff_Z) * 100)
    plt.xlabel(r"t  ($\mu$s)")
    plt.ylabel(r"$\Delta  Z \ / \ Z_{max} \ (\%)$")
    plt.subplot(3, 4, 11)
    plt.plot(self2.time * 1e6, np.abs(diff_phi))
    plt.xlabel(r"t  ($\mu$s)")
    plt.ylabel(r"$\Delta \Phi \ / \ (2 \pi)$")

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    if show:
        plt.show()
    plt.close()
