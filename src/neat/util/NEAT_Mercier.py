#!/usr/bin/env python3
###
# Driver script to run gyronimo's NEAT python's wrapper
# Uses qsc code as a stellarator repository and iota, G0, G2 and beta_1s calculator
# Rogerio Jorge, July 2021, Greifswald
###

import math
import time

import matplotlib.pyplot as plt
import neatpp.NEAT_Mercier as NEAT_Mercier
import numpy as np
from qsc import Qsc

nphi = 251  # resolution of the magnetic axis

## Stellarator to analyze
lambda_current = 0.0
r0 = 0.15
Lambda = [0.2]
rc = [1, 0.045]
zs = [0, -0.045]
nfp = 3
etabar = 0.9
B0 = 3
stel = Qsc(
    rc=rc, zs=zs, etabar=etabar, nfp=nfp, I2=lambda_current * B0 / 2, nphi=nphi, B0=B0
)
# B0 = 3
# r0=0.15
# lambda_current = 0.0
# Lambda   = [0.1]
# Tfinal   = 400
# stel = Qsc.from_paper(4, nphi=nphi, I2=lambda_current*B0/2)
# theta0   = [1.15]
# phi0     = [0.0]

s_of_phi = stel.phi * stel.axis_length / (2 * np.pi)
F0 = stel.iotaN / (B0 * stel.axis_length)
phi2c = (B0 / 2) * (
    2 * np.pi * B0 * F0 * stel.sigma
    - np.matmul(stel.d_d_phi, stel.curvature) / (stel.curvature * stel.d_l_d_phi)
)
phi2s = (B0 / 2) * (
    -(
        2
        * np.pi
        * B0
        * F0
        * stel.etabar
        * stel.etabar
        / (stel.curvature * stel.curvature)
        + stel.torsion
    )
)

## Initial conditions
theta0 = [1.14]
phi0 = [0.0]
charge = 1
rhom = 1
mass = 1
energy = [1e5]
nsamples = 5000
Tfinal = 500

showPlots = 1


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


def orbit(stel, r0, theta0, phi0, charge, rhom, mass, Lambda, energy, nsamples, Tfinal):
    # Call Gyronimo
    sol = np.array(
        NEAT_Mercier.gc_solver_Mercier(
            int(stel.nfp),
            stel.Bbar,
            lambda_current,
            np.append(s_of_phi, stel.axis_length),
            np.append(stel.curvature, stel.curvature[0]),
            np.append(stel.torsion, stel.torsion[0]),
            [B0] * nphi,
            np.append(phi2c, phi2c[0]),
            np.append(phi2s, phi2s[0]),
            charge,
            rhom,
            mass,
            Lambda,
            energy,
            r0,
            theta0,
            phi0,
            nsamples,
            Tfinal,
        )
    )

    # Store all output quantities
    time = sol[:, 0]
    r_pos = sol[:, 1]
    theta_pos = sol[:, 2]
    varphi = sol[:, 3] * 2 * np.pi / stel.axis_length
    from scipy.interpolate import CubicSpline as spline

    nu = stel.varphi - stel.phi
    nu_spline_of_varphi = spline(
        np.append(stel.varphi, 2 * np.pi / stel.nfp),
        np.append(nu, nu[0]),
        bc_type="periodic",
    )
    phi_pos = varphi - nu_spline_of_varphi(varphi)
    energy_parallel = sol[:, 4]
    energy_perpendicular = sol[:, 5]
    total_energy = energy_parallel + energy_perpendicular
    Bfield = sol[:, 6]
    v_parallel = sol[:, 7]
    rdot = sol[:, 8]
    thetadot = sol[:, 9]
    phidot = sol[:, 10]
    vppdot = sol[:, 11]

    # Calculate canonical angular momentum p_phi
    m_proton = 1.67262192369e-27
    e = 1.602176634e-19
    mu0 = 1.25663706212e-6
    Valfven = stel.Bbar / np.sqrt(mu0 * rhom * m_proton * 1.0e19)
    try:
        stel.G2
    except:
        stel.G2 = 0
    p_phi1 = (
        mass
        * m_proton
        * v_parallel
        * Valfven
        * (stel.G0 + r_pos**2 * (stel.G2 + (stel.iota - stel.iotaN) * stel.I2))
        / Bfield
        / stel.Bbar
    )
    p_phi2 = charge * e * r_pos**2 * stel.Bbar / 2 * stel.iotaN
    p_phi = p_phi1 - p_phi2
    return [
        time,
        r_pos,
        theta_pos,
        phi_pos,
        total_energy,
        theta0,
        phi0,
        Lambda,
        energy,
        p_phi,
        rdot,
        thetadot,
        phidot,
        vppdot,
        v_parallel,
        Bfield,
    ]


if __name__ == "__main__":
    print("---------------------------------")
    if stel.iotaN - stel.iota == 0:
        print("Quasi-axisymmetric stellarator with iota =", stel.iota)
    else:
        print("Quasi-helically symmetric stellarator with iota =", stel.iota)
    print("---------------------------------")
    print("Get particle orbit using gyronimo")
    result = []
    start_time = time.time()
    for _phi0 in phi0:
        for _Lambda in Lambda:
            for _energy in energy:
                for _theta0 in theta0:
                    orbit_temp = orbit(
                        stel,
                        r0,
                        _theta0,
                        _phi0,
                        charge,
                        rhom,
                        mass,
                        _Lambda,
                        _energy,
                        nsamples,
                        Tfinal,
                    )
                    if not math.isnan(orbit_temp[1][-1]):
                        result.append(orbit_temp)
    print("--- %s seconds ---" % (time.time() - start_time))

    # Check energy error for each orbit
    energy_error = [
        np.log10(np.abs(res[4] - res[4][0] + 1e-30) / res[4][0]) for res in result
    ]
    max_energy_error = [max(eerror[3::]) for eerror in energy_error]
    print("Max Energy Error per orbit = ", max_energy_error)

    # Check canonical angular momentum error for each orbit
    p_phi_error = [
        np.log10(np.abs(res[9] - res[9][0] + 1e-30) / abs(res[9][0])) for res in result
    ]
    max_p_phi_error = [max(perror[3::]) for perror in p_phi_error]
    print("Max Canonical Angular Momentum Error per orbit = ", max_p_phi_error)

    # Plot relevant quantities
    if showPlots == 1:
        fig = plt.figure(figsize=(10, 6))
        plt.subplot(3, 3, 1)
        [plt.plot(res[0], res[1]) for res in result]
        plt.xlabel("Time")
        plt.ylabel("r")
        plt.subplot(3, 3, 2)
        [plt.plot(res[0], res[4]) for res in result]
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.subplot(3, 3, 3)
        [plt.plot(res[0], res[9]) for res in result]
        plt.xlabel("Time")
        plt.ylabel("p_phi")
        plt.subplot(3, 3, 4)
        [plt.plot(res[0], res[3]) for res in result]
        plt.xlabel("Time")
        plt.ylabel("Phi")
        plt.subplot(3, 3, 5)
        [plt.plot(res[0], res[2]) for res in result]
        plt.xlabel("Time")
        plt.ylabel("Theta")
        plt.subplot(3, 3, 6)
        [plt.plot(res[0], res[14]) for res in result]
        plt.xlabel("Time")
        plt.ylabel("V_parallel")

        plt.subplot(3, 3, 7)
        [plt.plot(res[1] * np.cos(res[2]), res[1] * np.sin(res[2])) for res in result]
        th = np.linspace(0, 2 * np.pi, 100)
        plt.plot(r0 * np.cos(th), r0 * np.sin(th))
        plt.xlabel("r cos(theta)")
        plt.ylabel("r sin(theta)")

        rpos_cylindrical = [
            [stel.R0_func(res[3]), 0 * res[3], stel.Z0_func(res[3])]
            + res[1]
            * (
                np.cos(res[2])
                * [
                    stel.normal_R_spline(res[3]),
                    stel.normal_phi_spline(res[3]),
                    stel.normal_z_spline(res[3]),
                ]
                + np.sin(res[2])
                * [
                    stel.binormal_R_spline(res[3]),
                    stel.binormal_phi_spline(res[3]),
                    stel.binormal_z_spline(res[3]),
                ]
            )
            for res in result
        ]
        rpos_cartesian = [
            [
                rpos_cylindrical[i][0] * np.cos(result[i][3])
                - rpos_cylindrical[i][1] * np.sin(result[i][3]),
                rpos_cylindrical[i][0] * np.sin(result[i][3])
                + rpos_cylindrical[i][1] * np.cos(result[i][3]),
                rpos_cylindrical[i][2],
            ]
            for i in range(len(result))
        ]
        boundary = np.array(
            stel.get_boundary(
                r=0.9 * r0, nphi=90, ntheta=25, ntheta_fourier=16, mpol=8, ntor=15
            )
        )

        plt.subplot(3, 3, 8)
        [
            plt.plot(rpos_cylindrical[i][0], rpos_cylindrical[i][2])
            for i in range(len(result))
        ]
        phi1dplot_RZ = np.linspace(0, 2 * np.pi / stel.nfp, 4, endpoint=False)
        [
            plt.plot(
                boundary[3, :, int(phi / (2 * np.pi) * 90)],
                boundary[2, :, int(phi / (2 * np.pi) * 90)],
            )
            for phi in phi1dplot_RZ
        ]
        plt.xlabel("R")
        plt.ylabel("Z")

        curvature_spline = stel.convert_to_spline(stel.curvature)
        plt.subplot(3, 3, 9)
        [plt.plot(res[0], res[15], label="gyronimo") for res in result]
        [
            plt.plot(
                res[0],
                B0 / (1 + res[1] * curvature_spline(res[3]) * np.cos(res[2])),
                label="theory",
            )
            for res in result
        ]
        plt.xlabel("Time")
        plt.ylabel("Bfield")
        plt.legend()

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        [
            ax.plot3D(rpos_cartesian[i][0], rpos_cartesian[i][1], rpos_cartesian[i][2])
            for i in range(len(result))
        ]
        ax.plot_surface(boundary[0], boundary[1], boundary[2], alpha=0.5)
        set_axes_equal(ax)
        ax.set_axis_off()
        ax.dist = 6.0

        # import mayavi.mlab as mlab
        # fig = mlab.figure(bgcolor=(1,1,1), size=(430,720))
        # [mlab.plot3d(rpos_cartesian[i][0],rpos_cartesian[i][1],rpos_cartesian[i][2], color=(0.5,0.5,0.5)) for i in range(len(result))]

        # ntheta=80
        # nphi=220
        # X_qsc, Y_qsc, Z_qsc, R_qsc = stel.get_boundary(r=0.95*r0, ntheta=ntheta, nphi=nphi)

        # theta1D = np.linspace(0, 2 * np.pi, ntheta)
        # phi1D = np.linspace(0, 2 * np.pi, nphi)
        # phi2D, theta2D = np.meshgrid(phi1D, theta1D)
        # Bmag = stel.B_mag(r0, theta2D, phi2D)

        # mlab.mesh(X_qsc, Y_qsc, Z_qsc, scalars=Bmag, colormap='viridis')
        # mlab.view(azimuth=0, elevation=0, distance=8.5, focalpoint=(-0.15,0,0), figure=fig)

        # mlab.show()
        plt.show()
