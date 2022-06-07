#!/usr/bin/env python3
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from qsc import Qsc
from scipy.interpolate import UnivariateSpline as spline

## Stellarator to analyze
eR = [1, 0.0001, 0.0]  # radial cos coefficients
eZ = [0, 0.0001, 0.0]  # vertical sin coefficients
eta = 1.05  # quasisymmetry parameter
NFP = 1  # number of field periods
r0 = 0.4

# eR  = [ 1.2203214015975046,0.2066691989039873,0.020308603720394853,0.0003542260844918686,-0.0007076840679365638,-0.00021214649059550595 ]
# eZ  = [ 0.0,-0.16662936108007992,-0.021679470445210847,-0.0024273477846364767,-0.0004169761874001585,-5.249943146427062e-05 ]
# eta = 1.331610781926116
# NFP = 4
# r0  = 0.1

# eR  = [3.75925, 0.00766231, 0.0000175222, -1.20564e-6]
# eZ  = [0., -0.00778911, -0.0000129754, 1.09182e-6]
# NFP = 10
# eta = 0.25
# r0  = 0.9

## Metric Verification parameters
n_verifyMetric = 1  # 1 - verify metric and its derivatives, 0 - don't
# Resolution scan
NphiVec = [10, 30, 100, 300, 1000, 3000, 10000]
# Points to average over
pointVar = 0.35
npoints = 100
points = np.asarray(npoints * [[0.31241, 0.62312, 0.341231]])
points += pointVar * (np.random.rand(*points.shape) - 0.5)
## Particle Orbit parameters
nphiOrbit = 400
theta_initial = [3.0]  # [5.0,5.2,5.4,5.6,5.8,6.0]
phi_initial = [3.0]  # [5.0,5.2,5.4,5.6,5.8,6.0]
B0 = 1.0
charge = 1
rhom = 1
mass = 1
Lambda = 0.6
energy = 10000
nsamples = 50000
tfinal = 500

## Obtain Stellarator
stel = Qsc(rc=eR, zs=eZ, etabar=eta, nfp=NFP, nphi=nphiOrbit, I2=0.6)


def Raxisf(phi):
    return sum([eR[i] * np.cos(i * NFP * phi) for i in range(len(eR))])


def Zaxisf(phi):
    return sum([eZ[i] * np.sin(i * NFP * phi) for i in range(len(eZ))])


def sigma(phi):
    sp = spline(stel.phi, stel.sigma, k=3, s=0)
    return sp(np.mod(phi, 2 * np.pi / NFP))


def curvature(phi):
    sp = spline(stel.phi, stel.curvature, k=3, s=0)
    return sp(np.mod(phi, 2 * np.pi / NFP))


def normalR(phi):
    sp = spline(stel.phi, stel.normal_cylindrical[:, 0], k=3, s=0)
    return sp(np.mod(phi, 2 * np.pi / NFP))


def normalZ(phi):
    sp = spline(stel.phi, stel.normal_cylindrical[:, 2], k=3, s=0)
    return sp(np.mod(phi, 2 * np.pi / NFP))


def binormalR(phi):
    sp = spline(stel.phi, stel.binormal_cylindrical[:, 0], k=3, s=0)
    return sp(np.mod(phi, 2 * np.pi / NFP))


def binormalZ(phi):
    sp = spline(stel.phi, stel.binormal_cylindrical[:, 2], k=3, s=0)
    return sp(np.mod(phi, 2 * np.pi / NFP))


# def getFourierCurve(outputFile,ppp=10):
# 	from simsopt.geo.curvexyzfourier import CurveXYZFourier
# 	from simsopt.core.optimizable import optimizable
# 	coil_data = np.loadtxt(outputFile, delimiter=',')
# 	Nt_coils=len(coil_data)-1
# 	num_coils = int(len(coil_data[0])/6)
# 	coils = [optimizable(CurveXYZFourier(Nt_coils*ppp, Nt_coils)) for i in range(num_coils)]
# 	for ic in range(num_coils):
# 		dofs = coils[ic].dofs
# 		dofs[0][0] = coil_data[0, 6*ic + 1]
# 		dofs[1][0] = coil_data[0, 6*ic + 3]
# 		dofs[2][0] = coil_data[0, 6*ic + 5]
# 		for io in range(0, Nt_coils):
# 			dofs[0][2*io+1] = coil_data[io+1, 6*ic + 0]
# 			dofs[0][2*io+2] = coil_data[io+1, 6*ic + 1]
# 			dofs[1][2*io+1] = coil_data[io+1, 6*ic + 2]
# 			dofs[1][2*io+2] = coil_data[io+1, 6*ic + 3]
# 			dofs[2][2*io+1] = coil_data[io+1, 6*ic + 4]
# 			dofs[2][2*io+2] = coil_data[io+1, 6*ic + 5]
# 		coils[ic].set_dofs(np.concatenate(dofs))
# 	return coils


def plot_stellarator(ax, rR, ntheta=50, nphi=200):
    def X1cF(phi):
        sp = spline(stel.phi, stel.X1c, k=3, s=0)
        return sp(np.mod(phi, 2 * np.pi / NFP))

    def X1sF(phi):
        sp = spline(stel.phi, stel.X1s, k=3, s=0)
        return sp(np.mod(phi, 2 * np.pi / NFP))

    def Y1cF(phi):
        sp = spline(stel.phi, stel.Y1c, k=3, s=0)
        return sp(np.mod(phi, 2 * np.pi / NFP))

    def Y1sF(phi):
        sp = spline(stel.phi, stel.Y1s, k=3, s=0)
        return sp(np.mod(phi, 2 * np.pi / NFP))

    # Surface components in cylindrical coordinates
    def rSurf(r, phi, theta):
        return (
            Raxisf(phi)
            + r * X1cF(phi) * np.cos(theta - NFP * phi) * normalR(phi)
            + r * X1sF(phi) * np.sin(theta - NFP * phi) * normalR(phi)
            + r * Y1cF(phi) * np.cos(theta - NFP * phi) * binormalR(phi)
            + r * Y1sF(phi) * np.sin(theta - NFP * phi) * binormalR(phi)
        )

    def zSurf(r, phi, theta):
        return (
            Zaxisf(phi)
            + r * X1cF(phi) * np.cos(theta - NFP * phi) * normalZ(phi)
            + r * X1sF(phi) * np.sin(theta - NFP * phi) * normalZ(phi)
            + r * Y1cF(phi) * np.cos(theta - NFP * phi) * binormalZ(phi)
            + r * Y1sF(phi) * np.sin(theta - NFP * phi) * binormalZ(phi)
        )

    def Bf(r, phi, theta):
        return stel.B0 * (1 + r * eta * np.cos(theta - NFP * phi))

    theta1D = np.linspace(0, 2 * np.pi, ntheta)
    phi1D = np.linspace(0, 2 * np.pi, nphi)
    Xsurf = np.zeros((ntheta, nphi))
    Ysurf = np.zeros((ntheta, nphi))
    Zsurf = np.zeros((ntheta, nphi))
    Bmag = np.zeros((ntheta, nphi))
    for countT, th in enumerate(theta1D):
        for countP, ph in enumerate(phi1D):
            rs = rSurf(rR, ph, th)
            zs = zSurf(rR, ph, th)
            Xsurf[countT, countP] = rs * np.cos(ph)
            Ysurf[countT, countP] = rs * np.sin(ph)
            Zsurf[countT, countP] = zs
            Bmag[countT, countP] = Bf(rR, ph, th)
    B_rescaled = (Bmag - Bmag.min()) / (Bmag.max() - Bmag.min())
    ax.plot_surface(
        Xsurf,
        Ysurf,
        Zsurf,
        facecolors=cm.jet(B_rescaled),
        rstride=1,
        cstride=1,
        antialiased=False,
        linewidth=0,
        alpha=0.25,
    )
    ax.auto_scale_xyz(
        [Xsurf.min(), Xsurf.max()],
        [Xsurf.min(), Xsurf.max()],
        [Xsurf.min(), Xsurf.max()],
    )
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)

    # coils = getFourierCurve("/Users/rogeriojorge/Dropbox/SACOPT/OptAxis_coil_coeffs.dat")
    # gamma = coils[0].gamma()
    # N = gamma.shape[0]
    # l = len(coils)
    # data = np.zeros((l*(N+1), 3))
    # for i in range(l):
    #     data[(i*(N+1)):((i+1)*(N+1)-1), :] = coils[i].gamma()
    #     data[((i+1)*(N+1)-1), :] = coils[i].gamma()[0, :]
    # ax.scatter(data[:,0], data[:,1], data[:,2], '-',s=0.2,linewidth=0.2,color='black',alpha=0.6)

    plt.tight_layout()
    mlab.mesh(
        Xsurf,
        Ysurf,
        Zsurf,
        opacity=0.2,
        scalars=None,
        line_width=0.0,
        color=(0, 0, 0),
        representation="wireframe",
    )
    # mlab.points3d(data[:,0], data[:,1], data[:,2], color=(0.58, 0.33, 0.01), mode='sphere', scale_factor=0.02)


def particleOrbit(ax, nphi, r0, theta_initial, phi_initial):
    ## Obtain particle orbit
    sol = np.array(
        helna.gc_solver(
            stel.rc[0],
            int(len(stel.rc) - 1),
            stel.rc[1::],
            stel.zs[1::],
            stel.etabar,
            stel.iota,
            stel.iota - stel.iotaN,
            stel.axis_length,
            NFP,
            nphi,
            stel.phi,
            stel.sigma,
            stel.curvature,
            stel.torsion,
            stel.d_l_d_phi,
            points,
            B0,
            charge,
            rhom,
            mass,
            Lambda,
            energy,
            r0,
            theta_initial,
            phi_initial,
            nsamples,
            tfinal,
        ),
        dtype=np.float64,
    )

    time = sol[:, 0]
    r_pos = sol[:, 1]
    theta_pos = sol[:, 2]
    phi_pos = sol[:, 3]
    energy_parallel = sol[:, 4]
    energy_perpendicular = sol[:, 5]
    modB = sol[:, 6]

    total_energy = energy_parallel + energy_perpendicular
    energy_error = np.log10(
        np.abs(total_energy - total_energy[0] + 1e-30) / total_energy[0]
    )
    print(
        "Energy log10 error: max =",
        max(energy_error[2::]),
        "min =",
        min(energy_error[2::]),
    )

    # unknown_pos = sol[:,4]
    # t_pos     = sol[:,0]
    # plt.plot(t_pos,r_pos-r0)
    # plt.xlabel('Time', fontsize=18)
    # plt.ylabel('Radius-Radius(initial)', fontsize=16)
    # plt.tight_layout()
    # plt.savefig('Radius_vs_Time.png')

    xParticle = np.zeros((len(r_pos),))
    yParticle = np.zeros((len(r_pos),))
    zParticle = np.zeros((len(r_pos),))
    for i in range(len(r_pos)):
        rsurf = (
            Raxisf(phi_pos[i])
            + r_pos[i]
            * eta
            / curvature(phi_pos[i])
            * np.cos(theta_pos[i] - NFP * phi_pos[i])
            * normalR(phi_pos[i])
            + r_pos[i]
            / eta
            * curvature(phi_pos[i])
            * (
                np.sin(theta_pos[i] - NFP * phi_pos[i])
                + sigma(phi_pos[i]) * np.cos(theta_pos[i] - NFP * phi_pos[i])
            )
            * binormalR(phi_pos[i])
        )
        zsurf = (
            Zaxisf(phi_pos[i])
            + r_pos[i]
            * eta
            / curvature(phi_pos[i])
            * np.cos(theta_pos[i] - NFP * phi_pos[i])
            * normalZ(phi_pos[i])
            + r_pos[i]
            / eta
            * curvature(phi_pos[i])
            * (
                np.sin(theta_pos[i] - NFP * phi_pos[i])
                + sigma(phi_pos[i]) * np.cos(theta_pos[i] - NFP * phi_pos[i])
            )
            * binormalZ(phi_pos[i])
        )
        xParticle[i] = rsurf * np.cos(phi_pos[i])
        yParticle[i] = rsurf * np.sin(phi_pos[i])
        zParticle[i] = zsurf

    mlab.plot3d(
        xParticle, yParticle, zParticle, color=(1.0, 0.0, 0.0), tube_radius=0.005
    )
    ax.scatter(
        xParticle, yParticle, zParticle, marker=".", s=1.5, linewidths=0.0, color="red"
    )


def verifyMetric(count):
    ## Obtain Stellna's positions, metric and derivatives
    [positions, gHelna, dgHelna] = helna.metric_info(
        stel.rc[0],
        int(len(stel.rc) - 1),
        stel.rc[1::],
        stel.zs[1::],
        stel.etabar,
        stel.iota,
        stel.iota - stel.iotaN,
        stel.axis_length,
        NFP,
        NphiVec[count],
        stel.phi,
        stel.sigma,
        stel.curvature,
        stel.torsion,
        stel.d_l_d_phi,
        stel.d_nu1c_d_phi,
        stel.d_nu1s_d_phi,
        points,
    )

    ## Verify that points are exactly the same
    assert np.allclose(points, positions)

    ## Define metric and derivatives
    def curvature(phi):
        sp = spline(stel.phi, stel.curvature, k=3, s=0)
        return sp(np.mod(phi, 2 * np.pi / NFP))

    def dcurvature(phi):
        sp = spline(stel.phi, stel.curvature, k=3, s=0).derivative(n=1)
        return sp(np.mod(phi, 2 * np.pi / NFP))

    def ddcurvature(phi):
        sp = spline(stel.phi, stel.curvature, k=3, s=0).derivative(n=2)
        return sp(np.mod(phi, 2 * np.pi / NFP))

    def torsion(phi):
        sp = spline(stel.phi, stel.torsion, k=3, s=0)
        return sp(np.mod(phi, 2 * np.pi / NFP))

    def dtorsion(phi):
        sp = spline(stel.phi, stel.torsion, k=3, s=0).derivative(n=1)
        return sp(np.mod(phi, 2 * np.pi / NFP))

    def sigma(phi):
        sp = spline(stel.phi, stel.sigma, k=3, s=0)
        return sp(np.mod(phi, 2 * np.pi / NFP))

    def dsigma(phi):
        sp = spline(stel.phi, stel.sigma, k=3, s=0).derivative(n=1)
        return sp(np.mod(phi, 2 * np.pi / NFP))

    def ddsigma(phi):
        sp = spline(stel.phi, stel.sigma, k=3, s=0).derivative(n=2)
        return sp(np.mod(phi, 2 * np.pi / NFP))

    def d_l_d_phi(phi):
        sp = spline(stel.phi, stel.d_l_d_phi, k=3, s=0)
        return sp(np.mod(phi, 2 * np.pi / NFP))

    def dd_l_d_phi(phi):
        sp = spline(stel.phi, stel.d_l_d_phi, k=3, s=0).derivative(n=1)
        return sp(np.mod(phi, 2 * np.pi / NFP))

    def grr(r, theta, phi):
        return (
            np.cos(theta) ** 2 * eta**2 / curvature(phi) ** 2
            + (np.sin(theta) + np.cos(theta) * sigma(phi)) ** 2
            * curvature(phi) ** 2
            / eta**2
        )

    def gtt(r, theta, phi):
        return (
            r**2
            * (
                eta**4 * np.sin(theta) ** 2
                + curvature(phi) ** 4
                * (np.cos(theta) - np.sin(theta) * sigma(phi)) ** 2
            )
            / eta**2
            / curvature(phi) ** 2
        )

    def gpp(r, theta, phi):
        coso = np.cos(theta)
        sino = np.sin(theta)
        return (
            (-1 + coso * r * (eta)) ** 2 * (d_l_d_phi(phi)) ** 2
            + (
                (coso * r * (eta) * (dcurvature(phi))) / curvature(phi) ** 2
                + (
                    curvature(phi)
                    * r
                    * (coso * sigma(phi) + sino)
                    * torsion(phi)
                    * (d_l_d_phi(phi))
                )
                / (eta)
            )
            ** 2
            + (
                (coso * r * torsion(phi) * (eta) * (d_l_d_phi(phi))) / curvature(phi)
                + (
                    r
                    * (
                        (coso * sigma(phi) + sino) * (dcurvature(phi))
                        + coso * curvature(phi) * (dsigma(phi))
                    )
                )
                / (eta)
            )
            ** 2
        )

    def grt(r, theta, phi):
        return (
            -r * eta**2 * np.cos(theta) * np.sin(theta) / curvature(phi) ** 2
            + r
            * (np.sin(theta) + np.cos(theta) * sigma(phi))
            * (np.cos(theta) - np.sin(theta) * sigma(phi))
            * curvature(phi) ** 2
            / eta**2
        )

    def grp(r, theta, phi):
        return (r * curvature(phi) ** 2 / eta**2) * (
            (
                -(eta**4) * np.cos(theta) ** 2 / curvature(phi) ** 4
                + (np.sin(theta) + np.cos(theta) * sigma(phi)) ** 2
            )
            * dcurvature(phi)
            / curvature(phi)
            + np.cos(theta) * (np.sin(theta) + np.cos(theta) * sigma(phi)) * dsigma(phi)
        )

    def gtp(r, theta, phi):
        return (r**2 / 2 / eta**2 / curvature(phi) ** 3) * (
            2 * eta**2 * curvature(phi) ** 3 * torsion(phi) * d_l_d_phi(phi)
            + eta**4 * np.sin(2 * theta) * dcurvature(phi)
            - curvature(phi) ** 4
            * (
                -np.sin(2 * theta)
                - 2 * np.cos(2 * theta) * sigma(phi)
                + np.sin(2 * theta) * sigma(phi) ** 2
            )
            * dcurvature(phi)
            + 2
            * np.cos(theta)
            * curvature(phi) ** 5
            * (np.cos(theta) - np.sin(theta) * sigma(phi))
            * dsigma(phi)
        )

    def dgrr(r, theta, phi):
        d_r_grr = 0
        d_theta_grr = (
            -2 * np.cos(theta) * np.sin(theta) * eta**2 / curvature(phi) ** 2
            + 2
            * (np.sin(theta) + np.cos(theta) * sigma(phi))
            * (np.cos(theta) - np.sin(theta) * sigma(phi))
            * curvature(phi) ** 2
            / eta**2
        )
        d_phi_grr = (
            -2 * dcurvature(phi) * np.cos(theta) ** 2 * eta**2 / curvature(phi) ** 3
            + (np.sin(theta) + np.cos(theta) * sigma(phi)) ** 2
            * dcurvature(phi)
            * 2
            * curvature(phi)
            / eta**2
            + 2
            * np.cos(theta)
            * dsigma(phi)
            * (np.sin(theta) + np.cos(theta) * sigma(phi))
            * curvature(phi) ** 2
            / eta**2
        )
        return [d_r_grr, d_theta_grr, d_phi_grr]

    def dgtt(r, theta, phi):
        d_r_gtt = (
            2
            * r
            * (
                eta**4 * np.sin(theta) ** 2
                + curvature(phi) ** 4
                * (np.cos(theta) - np.sin(theta) * sigma(phi)) ** 2
            )
            / eta**2
            / curvature(phi) ** 2
        )
        d_theta_gtt = (
            r**2
            * (
                eta**4 * 2 * np.sin(theta) * np.cos(theta)
                + curvature(phi) ** 4
                * (np.cos(theta) - np.sin(theta) * sigma(phi))
                * 2
                * (-np.sin(theta) - np.cos(theta) * sigma(phi))
            )
            / eta**2
            / curvature(phi) ** 2
        )
        d_phi_gtt = (
            r**2
            * (
                -2 * dcurvature(phi) / curvature(phi) * eta**4 * np.sin(theta) ** 2
                + 2
                * curvature(phi) ** 3
                * dcurvature(phi)
                * (np.cos(theta) - np.sin(theta) * sigma(phi)) ** 2
                + curvature(phi) ** 4
                * (np.cos(theta) - np.sin(theta) * sigma(phi))
                * 2
                * (-np.sin(theta) * dsigma(phi))
            )
            / eta**2
            / curvature(phi) ** 2
        )
        return [d_r_gtt, d_theta_gtt, d_phi_gtt]

    def dgpp(r, theta, phi):
        coso = np.cos(theta)
        sino = np.sin(theta)
        d_r_gpp = (
            2
            * (
                coso * d_l_d_phi(phi) ** 2 * eta**3 * (-1 + coso * eta * r)
                + (
                    r
                    * (
                        curvature(phi)
                        * (
                            coso * curvature(phi) * dsigma(phi)
                            + dcurvature(phi) * (coso * sigma(phi) + sino)
                        )
                        + coso * d_l_d_phi(phi) * eta**2 * torsion(phi)
                    )
                    ** 2
                )
                / curvature(phi) ** 2
                + (
                    r
                    * (
                        coso * dcurvature(phi) * eta**2
                        + curvature(phi) ** 3
                        * d_l_d_phi(phi)
                        * (coso * sigma(phi) + sino)
                        * torsion(phi)
                    )
                    ** 2
                )
                / curvature(phi) ** 4
            )
        ) / eta**2
        d_theta_gpp = (
            2
            * r
            * (
                -(d_l_d_phi(phi) ** 2 * eta**3 * (-1 + coso * eta * r) * sino)
                - (
                    r
                    * (
                        curvature(phi)
                        * (
                            coso * curvature(phi) * dsigma(phi)
                            + dcurvature(phi) * (coso * sigma(phi) + sino)
                        )
                        + coso * d_l_d_phi(phi) * eta**2 * torsion(phi)
                    )
                    * (
                        curvature(phi)
                        * (
                            curvature(phi) * dsigma(phi) * sino
                            + dcurvature(phi) * (-coso + sigma(phi) * sino)
                        )
                        + d_l_d_phi(phi) * eta**2 * sino * torsion(phi)
                    )
                )
                / curvature(phi) ** 2
                - (
                    r
                    * (
                        coso * dcurvature(phi) * eta**2
                        + curvature(phi) ** 3
                        * d_l_d_phi(phi)
                        * (coso * sigma(phi) + sino)
                        * torsion(phi)
                    )
                    * (
                        dcurvature(phi) * eta**2 * sino
                        + curvature(phi) ** 3
                        * d_l_d_phi(phi)
                        * (-coso + sigma(phi) * sino)
                        * torsion(phi)
                    )
                )
                / curvature(phi) ** 4
            )
        ) / eta**2
        d_phi_gpp = 2 * (
            d_l_d_phi(phi) * (-1 + coso * eta * r) ** 2 * (dd_l_d_phi(phi))
            + (
                r**2
                * (
                    curvature(phi)
                    * (
                        coso * curvature(phi) * dsigma(phi)
                        + dcurvature(phi) * (coso * sigma(phi) + sino)
                    )
                    + coso * d_l_d_phi(phi) * eta**2 * torsion(phi)
                )
                * (
                    curvature(phi)
                    * (
                        coso * curvature(phi) ** 2 * ddsigma(phi)
                        + coso * d_l_d_phi(phi) * dtorsion(phi) * eta**2
                        + curvature(phi)
                        * (
                            2 * coso * dcurvature(phi) * dsigma(phi)
                            + ddcurvature(phi) * (coso * sigma(phi) + sino)
                        )
                    )
                    - coso
                    * eta**2
                    * torsion(phi)
                    * (
                        dcurvature(phi) * d_l_d_phi(phi)
                        - curvature(phi) * (dd_l_d_phi(phi))
                    )
                )
            )
            / (curvature(phi) ** 3 * eta**2)
            + (
                r**2
                * (
                    coso * dcurvature(phi) * eta**2
                    + curvature(phi) ** 3
                    * d_l_d_phi(phi)
                    * (coso * sigma(phi) + sino)
                    * torsion(phi)
                )
                * (
                    -2 * coso * dcurvature(phi) ** 2 * eta**2
                    + coso * curvature(phi) * ddcurvature(phi) * eta**2
                    + curvature(phi) ** 3
                    * dcurvature(phi)
                    * d_l_d_phi(phi)
                    * (coso * sigma(phi) + sino)
                    * torsion(phi)
                    + curvature(phi) ** 4
                    * (
                        d_l_d_phi(phi) * dtorsion(phi) * (coso * sigma(phi) + sino)
                        + torsion(phi)
                        * (
                            coso * d_l_d_phi(phi) * dsigma(phi)
                            + (coso * sigma(phi) + sino) * (dd_l_d_phi(phi))
                        )
                    )
                )
            )
            / (curvature(phi) ** 5 * eta**2)
        )
        return [d_r_gpp, d_theta_gpp, d_phi_gpp]

    def dgrt(r, theta, phi):
        d_r_grt = (
            -(eta**2) * np.cos(theta) * np.sin(theta) / curvature(phi) ** 2
            + (np.sin(theta) + np.cos(theta) * sigma(phi))
            * (np.cos(theta) - np.sin(theta) * sigma(phi))
            * curvature(phi) ** 2
            / eta**2
        )
        d_theta_grt = (
            -r * eta**2 * np.cos(2 * theta) / curvature(phi) ** 2
            + r
            * (
                -2 * np.sin(2 * theta) * sigma(phi)
                + np.cos(2 * theta) * (1 - sigma(phi) ** 2)
            )
            * curvature(phi) ** 2
            / eta**2
        )
        d_phi_grt = (
            2
            * dcurvature(phi)
            * r
            * eta**2
            * np.cos(theta)
            * np.sin(theta)
            / curvature(phi) ** 3
            + 2
            * r
            * (np.sin(theta) + np.cos(theta) * sigma(phi))
            * (np.cos(theta) - np.sin(theta) * sigma(phi))
            * curvature(phi)
            * dcurvature(phi)
            / eta**2
            + r
            * (np.cos(2 * theta) - np.sin(2 * theta) * sigma(phi))
            * dsigma(phi)
            * curvature(phi) ** 2
            / eta**2
        )
        return [d_r_grt, d_theta_grt, d_phi_grt]

    def dgrp(r, theta, phi):
        d_r_grp = (curvature(phi) ** 2 / eta**2) * (
            (
                -(eta**4) * np.cos(theta) ** 2 / curvature(phi) ** 4
                + (np.sin(theta) + np.cos(theta) * sigma(phi)) ** 2
            )
            * dcurvature(phi)
            / curvature(phi)
            + np.cos(theta) * (np.sin(theta) + np.cos(theta) * sigma(phi)) * dsigma(phi)
        )
        d_theta_grp = (r / curvature(phi) ** 3 / eta**2) * (
            (
                2 * np.cos(2 * theta) * curvature(phi) ** 4 * sigma(phi)
                + np.sin(2 * theta)
                * (eta**4 - curvature(phi) ** 4 * (-1 + sigma(phi) ** 2))
            )
            * dcurvature(phi)
            + curvature(phi) ** 5
            * (np.cos(2 * theta) - np.sin(2 * theta) * sigma(phi))
            * dsigma(phi)
        )
        d_phi_grp = (r / curvature(phi) ** 4 / eta**2) * (
            (
                3 * eta**4 * np.cos(theta) ** 2
                + curvature(phi) ** 4
                * (np.sin(theta) + np.cos(theta) * sigma(phi)) ** 2
            )
            * dcurvature(phi) ** 2
            + 4
            * np.cos(theta)
            * curvature(phi) ** 5
            * (np.sin(theta) + np.cos(theta) * sigma(phi))
            * dcurvature(phi)
            * dsigma(phi)
            - eta**4 * np.cos(theta) ** 2 * curvature(phi) * ddcurvature(phi)
            + curvature(phi) ** 5
            * (np.sin(theta) + np.cos(theta) * sigma(phi)) ** 2
            * ddcurvature(phi)
            + np.cos(theta)
            * curvature(phi) ** 6
            * (
                np.cos(theta) * dsigma(phi) ** 2
                + (np.sin(theta) + np.cos(theta) * sigma(phi)) * ddsigma(phi)
            )
        )
        return [d_r_grp, d_theta_grp, d_phi_grp]

    def dgtp(r, theta, phi):
        d_r_gtp = (r * 2 / 2 / eta**2 / curvature(phi) ** 3) * (
            2 * eta**2 * curvature(phi) ** 3 * torsion(phi) * d_l_d_phi(phi)
            + eta**4 * np.sin(2 * theta) * dcurvature(phi)
            - curvature(phi) ** 4
            * (
                -np.sin(2 * theta)
                - 2 * np.cos(2 * theta) * sigma(phi)
                + np.sin(2 * theta) * sigma(phi) ** 2
            )
            * dcurvature(phi)
            + 2
            * np.cos(theta)
            * curvature(phi) ** 5
            * (np.cos(theta) - np.sin(theta) * sigma(phi))
            * dsigma(phi)
        )
        d_theta_gtp = (
            r**2
            * (
                (
                    -2 * np.sin(2 * theta) * curvature(phi) ** 4 * sigma(phi)
                    + np.cos(2 * theta)
                    * (eta**4 - curvature(phi) ** 4 * (-1 + sigma(phi) ** 2))
                )
                * dcurvature(phi)
                - curvature(phi) ** 5
                * (np.sin(2 * theta) + np.cos(2 * theta) * sigma(phi))
                * dsigma(phi)
            )
            / eta**2
            / curvature(phi) ** 3
        )
        d_phi_gtp = (
            r**2
            / 2
            / eta**2
            / curvature(phi) ** 4
            * (
                (
                    -3 * eta**4 * np.sin(2 * theta)
                    + curvature(phi) ** 4
                    * (
                        2 * np.cos(2 * theta) * sigma(phi)
                        - np.sin(2 * theta) * (-1 + sigma(phi) ** 2)
                    )
                )
                * dcurvature(phi) ** 2
                + 2
                * curvature(phi) ** 5
                * (1 + 2 * np.cos(2 * theta) - 2 * np.sin(2 * theta) * sigma(phi))
                * dcurvature(phi)
                * dsigma(phi)
                + 2
                * eta**2
                * curvature(phi) ** 4
                * (d_l_d_phi(phi) * dtorsion(phi) + torsion(phi) * dd_l_d_phi(phi))
                + eta**4 * np.sin(2 * theta) * curvature(phi) * ddcurvature(phi)
                + curvature(phi) ** 5
                * (
                    2 * np.cos(2 * theta) * sigma(phi)
                    - np.sin(2 * theta) * (-1 + sigma(phi) ** 2)
                )
                * ddcurvature(phi)
                - 2
                * np.cos(theta)
                * curvature(phi) ** 6
                * (
                    np.sin(theta) * dsigma(phi) ** 2
                    + (-np.cos(theta) + np.sin(theta) * sigma(phi)) * ddsigma(phi)
                )
            )
        )
        return [d_r_gtp, d_theta_gtp, d_phi_gtp]

    def jacobian(r, theta, phi):
        return r * (1 - r * eta * np.cos(theta)) * d_l_d_phi(phi)

    ## Verify metric with Helena
    gQSC = np.array(
        [
            [
                grr(point[0], point[1], point[2]),
                gtt(point[0], point[1], point[2]),
                gpp(point[0], point[1], point[2]),
                grt(point[0], point[1], point[2]),
                grp(point[0], point[1], point[2]),
                gtp(point[0], point[1], point[2]),
            ]
            for point in points
        ],
        dtype=np.float64,
    )

    gDifferenceTemp = np.array(
        [
            [abs((gQSC[i, j] - gHelna[i][j]) / gHelna[i][j]) for j in range(6)]
            for i in range(len(points))
        ]
    )

    ## Verify metric derivatives with Helena
    dgQSC = np.array(
        [
            [
                dgrr(point[0], point[1], point[2])[0],
                dgrr(point[0], point[1], point[2])[1],
                dgrr(point[0], point[1], point[2])[2],
                dgtt(point[0], point[1], point[2])[0],
                dgtt(point[0], point[1], point[2])[1],
                dgtt(point[0], point[1], point[2])[2],
                dgpp(point[0], point[1], point[2])[0],
                dgpp(point[0], point[1], point[2])[1],
                dgpp(point[0], point[1], point[2])[2],
                dgrt(point[0], point[1], point[2])[0],
                dgrt(point[0], point[1], point[2])[1],
                dgrt(point[0], point[1], point[2])[2],
                dgrp(point[0], point[1], point[2])[0],
                dgrp(point[0], point[1], point[2])[1],
                dgrp(point[0], point[1], point[2])[2],
                dgtp(point[0], point[1], point[2])[0],
                dgtp(point[0], point[1], point[2])[1],
                dgtp(point[0], point[1], point[2])[2],
            ]
            for point in points
        ],
        dtype=np.float64,
    )

    dgDifferenceTemp = np.array(
        [
            [
                abs((dgQSC[i, j] - dgHelna[i][j]) / (dgHelna[i][j] + 1e-19))
                for j in range(18)
            ]
            for i in range(len(points))
        ]
    )

    ## Verify jacobian with Helene
    jacobQSC = np.array([jacobian(point[0], point[1], point[2]) for point in points])

    jacobDifferenceTemp = np.array(
        [abs((jacobQSC[i] - gHelna[i][6]) / gHelna[i][6]) for i in range(len(points))]
    )

    ## Return the percent difference
    return [gDifferenceTemp, dgDifferenceTemp, jacobDifferenceTemp]


if __name__ == "__main__":

    # fig = plt.figure()
    # fig.patch.set_facecolor('white')
    # ax = fig.gca(projection='3d')
    # import mayavi.mlab as mlab
    # mlab.options.offscreen = True # Show on screen or not
    # mlab.figure(bgcolor=(1,1,1),size=(2000,2000))
    # print("Plotting particles")
    # for i in range(len(theta_initial)):
    #     particleOrbit(ax,nphiOrbit,r0,theta_initial[i],phi_initial[i])
    # print("Plotting stellarator")
    # plot_stellarator(ax,r0)
    # plt.savefig('particle_gyronimo.png',dpi=500)
    # mlab.savefig("Particles.png", magnification='auto')
    # # mlab.show()
    # # plt.show()
    # mlab.close()

    if n_verifyMetric:
        ## Parallelization
        pool = multiprocessing.Pool(7)
        result = pool.map(verifyMetric, range(len(NphiVec)))
        ## Collect the resulting difference
        gDifference = np.array([np.array(result[:][i][0]) for i in range(len(NphiVec))])
        dDifference = np.array([np.array(result[:][i][1]) for i in range(len(NphiVec))])
        JDifference = np.array([np.array(result[:][i][2]) for i in range(len(NphiVec))])

        ## Average out all points
        gDifferenceAverage = np.average(gDifference, axis=1)
        dgDifferenceAverage = np.average(dDifference, axis=1)
        JDifferenceAverage = np.average(JDifference, axis=1)

        ## Plot metric
        fig = plt.figure()
        plt.plot(NphiVec, gDifferenceAverage)
        plt.plot(NphiVec, JDifferenceAverage)
        plt.xlabel("Phi Resolution")
        plt.ylabel("Relative Error")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(
            [
                r"$g_{rr}$",
                r"$g_{\theta \theta}$",
                r"$g_{\phi \phi}$",
                r"$g_{r \theta}$",
                r"$g_{r \phi}$",
                r"$g_{\theta \phi}$",
                r"$J$",
            ]
        )
        plt.savefig("metricVerification.png")
        # plt.show()

        ## Plot metric derivatives
        fig = plt.figure(figsize=[10, 10])
        plt.plot(NphiVec, dgDifferenceAverage)
        plt.xlabel("Phi Resolution")
        plt.ylabel("Relative Error")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(
            [
                r"$\partial_r g_{r r}$",
                r"$\partial_\theta g_{r r}$",
                r"$\partial_\phi g_{r r}$",
                r"$\partial_r g_{\theta \theta}$",
                r"$\partial_\theta g_{\theta \theta}$",
                r"$\partial_\phi g_{\theta \theta}$",
                r"$\partial_r g_{\phi \phi}$",
                r"$\partial_\theta g_{\phi \phi}$",
                r"$\partial_\phi g_{\phi \phi}$",
                r"$\partial_r g_{r \theta}$",
                r"$\partial_\theta g_{r \theta}$",
                r"$\partial_\phi g_{r \theta}$",
                r"$\partial_r g_{r \phi}$",
                r"$\partial_\theta g_{r \phi}$",
                r"$\partial_\phi g_{r \phi}$",
                r"$\partial_r g_{\theta \phi}$",
                r"$\partial_\theta g_{\theta \phi}$",
                r"$\partial_\phi g_{\theta \phi}$",
            ]
        )
        plt.savefig("dmetricVerification.png")
        plt.show()
