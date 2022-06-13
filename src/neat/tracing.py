""" Tracing module of NEAT

This script performs the tracing of particles by
connecting the user input to the gyronimo-based
functions defined in the neatpp.cpp file. It is
able to simulate both single particle and particle
ensemble orbits.

"""

import logging
import os
from pathlib import Path
from subprocess import run
from typing import Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline as spline
from scipy.interpolate import RectBivariateSpline, interp1d

from .constants import ELEMENTARY_CHARGE, PROTON_MASS
from .fields import Stellna, StellnaQS, Vmec
from .plotting import (
    plot_animation3d,
    plot_orbit2d,
    plot_orbit3d,
    plot_parameters,
    set_axes_equal,
)

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
        self, particle, field, nsamples=1000, tfinal=0.0001, constant_b20=False
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
            # Canonical angular momentum still not calculated for VMEC
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
        plot_orbit2d(
            x_position=self.r_pos * np.cos(self.theta_pos),
            y_position=self.r_pos * np.sin(self.theta_pos),
            show=show,
        )

    def plot_orbit_3d(self, r_surface=0.1, distance=6, show=True):
        """Plot particle orbit in 3D cartesian coordinates"""
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
            boundary = self.field.surface_xyz(
                iradius=self.field.nsurfaces - 1,
                ntheta=50,
                nzeta=int(90 * self.field.nfp),
            )

        plot_orbit3d(
            boundary=boundary,
            rpos_cartesian=self.rpos_cartesian,
            distance=distance,
            show=show,
        )

    def plot(self, show=True):
        """Plot relevant physics parameters of the particle orbit"""
        plot_parameters(self=self, show=show)

    def plot_animation(self, r_surface=0.1, distance=7, show=True, save_movie=False):
        """Plot three-dimensional animation of the particle orbit"""
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
            boundary = self.field.surface_xyz(
                iradius=self.field.nsurfaces - 1,
                ntheta=50,
                nzeta=int(90 * self.field.nfp),
            )

        plot_animation3d(
            boundary=boundary,
            rpos_cartesian=self.rpos_cartesian,
            nsamples=self.nsamples,
            distance=distance,
            show=show,
            save_movie=save_movie,
        )


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
        constant_b20=False,
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

    def plot_loss_fraction(self, show=True):
        """Make a plot of the fraction of total particles lost over time"""
        plt.semilogx(self.time, self.loss_fraction_array)
        plt.xlabel("Time (s)")
        plt.ylabel("Loss Fraction")
        plt.tight_layout()
        if show:
            plt.show()


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


class ParticleOrbitBeams3D:
    r"""
    Interface between user and BEAMS3D code.
    """

    def __init__(
        self,
        field: Vmec,
        results_folder: str = None,
        particle: ChargedParticle = ChargedParticle(),
        vmec_input_filename: str = None,
        nsamples=800,
        tfinal=0.0001,
        ntheta=60,
        nzeta=100,
        NR=32,
        NZ=32,
        NPHI=32,
    ) -> None:

        self.field = field
        self.particle = particle
        self.nsamples = nsamples

        # Obtain name of VMEC input/output and BEAMS3D output
        # Example: wout_name.nc yields name
        self.equilibrium_name = os.path.basename(field.wout_filename)[5:-3]

        # Create results folder if it doesn't exist
        if results_folder == None:
            results_folder = f"{os.getcwd()}/outputs"
        self.results_folder = results_folder
        Path(results_folder).mkdir(parents=True, exist_ok=True)

        # VMEC input file is the same as BEAMS3D input file
        if vmec_input_filename == None:
            vmec_input_filename = (
                f"{os.path.dirname(field.wout_filename)}/input.{self.equilibrium_name}"
            )

        # Compute VMEC outer surface (R,Z)
        r_surface, z_surface = field.surface_rz(field.nsurfaces - 1)

        # Compute initial particle position (R, Z, Phi)
        s_full = np.linspace(0, 1, field.nsurfaces)
        s_initial = particle.r_initial * particle.r_initial
        rc = interp1d(s_full, field.rmnc, axis=0, kind="linear")(s_initial)
        zs = interp1d(s_full, field.zmns, axis=0, kind="linear")(s_initial)
        theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=True)
        phi1d = np.linspace(0, 2 * np.pi / field.nfp, nzeta, endpoint=True)
        phi, theta = np.meshgrid(phi1d, theta1d)
        r = np.zeros((ntheta, nzeta))
        z = np.zeros((ntheta, nzeta))
        for imn in range(len(field.xm)):
            r += rc[imn] * np.cos(field.xm[imn] * theta - field.xn[imn] * phi)
            z += zs[imn] * np.sin(field.xm[imn] * theta - field.xn[imn] * phi)
        r_spl = RectBivariateSpline(theta1d, phi1d, r)
        z_spl = RectBivariateSpline(theta1d, phi1d, z)
        r_start = r_spl.ev([particle.theta_initial], [particle.phi_initial])[0]
        z_start = z_spl.ev([particle.theta_initial], [particle.phi_initial])[0]

        # Compute initial modulus of B
        ds = s_full[1] - s_full[0]
        s_half = s_full[1:] - ds / 2
        bc = interp1d(
            s_half, field.bmnc[1:, :], axis=0, kind="linear", fill_value="extrapolate"
        )(s_initial)
        modB = np.zeros((ntheta, nzeta))
        for imn in range(len(field.xm_nyq)):
            m = field.xm_nyq[imn]
            n = field.xn_nyq[imn]
            angle = m * theta - n * phi
            cosangle = np.cos(angle)
            modB += bc[imn] * cosangle
        modB_spl = RectBivariateSpline(theta1d, phi1d, modB)
        modB_initial = modB_spl.ev([particle.theta_initial], [particle.phi_initial])[0]

        # Compute initial particle velocity
        v_par_initial = (
            np.sign(particle.vpp_sign)
            * np.sqrt(
                2 * particle.energy * ELEMENTARY_CHARGE / (particle.mass * PROTON_MASS)
            )
            * (1 - particle.Lambda)
        )

        # Write BEAMS3D input
        beams3d_input = "&BEAMS3D_INPUT\n"
        beams3d_input += f"  R_START_IN = {r_start}\n"
        beams3d_input += f"  Z_START_IN = {z_start}\n"
        beams3d_input += f"  PHI_START_IN = {particle.phi_initial}\n"
        beams3d_input += f"  CHARGE_IN = {particle.charge * ELEMENTARY_CHARGE}\n"
        beams3d_input += f"  MASS_IN = {particle.mass * PROTON_MASS}\n"
        beams3d_input += f"  ZATOM_IN = {particle.charge}\n"
        beams3d_input += f"  T_END_IN = {tfinal}\n"
        beams3d_input += f"  NPOINC = {nsamples}\n"
        beams3d_input += f"  VLL_START_IN = {v_par_initial}\n"
        beams3d_input += f"  MU_START_IN = {particle.Lambda * particle.energy * ELEMENTARY_CHARGE / modB_initial}\n"
        beams3d_input += f"  NR = {NR}\n"
        beams3d_input += f"  NZ = {NZ}\n"
        beams3d_input += f"  NPHI = {NPHI}\n"
        beams3d_input += f"  RMIN = {np.min(r_surface)}\n"
        beams3d_input += f"  RMAX = {np.max(r_surface)}\n"
        beams3d_input += f"  ZMIN = {np.min(z_surface)}\n"
        beams3d_input += f"  ZMAX = {np.max(z_surface)}\n"
        beams3d_input += f"  PHIMIN =  0.\n"
        beams3d_input += f"  PHIMAX =  6.28318530718\n"
        beams3d_input += f"  INT_TYPE = 'LSODE'\n"
        beams3d_input += f"  FOLLOW_TOL =  1.E-12\n"
        beams3d_input += f"  VC_ADAPT_TOL =  1.E-3\n"

        with open(vmec_input_filename) as f:
            input_vmec = f.read()
        with open(vmec_input_filename, "a") as f:
            if "&BEAMS3D_INPUT" not in input_vmec:
                f.write(beams3d_input)
                f.write("/\n&END\n")
            else:
                input_vmec = input_vmec[0 : input_vmec.find("&BEAMS3D_INPUT")]
                f.truncate(0)
                f.write(input_vmec)
                f.write(beams3d_input)
                f.write("/\n&END\n")

    def run(self, beams3d_executable: str):
        """Run BEAMS3D"""
        self.beams3d_executable = beams3d_executable
        os.chdir(self.results_folder)
        bashCommand = (
            f"/.{self.beams3d_executable} -vmec {self.equilibrium_name} -plasma"
        )
        run(bashCommand.split())

    def plot(self, show: bool = True):
        """Plot the parameters of a particle orbit from BEAMS3D"""
        beams3d_file = f"{self.results_folder}/beams3d_{self.equilibrium_name}.h5"
        with h5py.File(beams3d_file, "r") as f:
            # To list all groups, run:
            # print("Keys: %s" % f.keys())
            B_lines = np.array(f["B_lines"])[0]
            NPOINC = np.array(f["npoinc"])[0]
            t_end = np.array(f["t_end"])[0]
            R_lines = np.array(f["R_lines"])[0]
            Z_lines = np.array(f["Z_lines"])[0]
            S_lines = np.array(f["S_lines"])[0]
            U_lines = np.array(f["U_lines"])[0]
            PHI_lines = np.array(f["PHI_lines"])[0]
            vll_lines = np.array(f["vll_lines"])[0]
            moment_lines = np.array(f["moment_lines"])[0]
        time = np.linspace(0, t_end, NPOINC + 1, endpoint=True)
        _ = plt.figure(figsize=(10, 6))
        plt.subplot(2, 4, 1)
        plt.plot(time, S_lines)
        plt.xlabel(r"$t (s)$")
        plt.ylabel(r"$s$")
        plt.subplot(2, 4, 2)
        plt.plot(time, U_lines)
        plt.xlabel(r"$t (s)$")
        plt.ylabel(r"$\theta$")
        plt.subplot(2, 4, 3)
        plt.plot(time, PHI_lines)
        plt.xlabel(r"$t (s)$")
        plt.ylabel(r"$phi$")
        plt.subplot(2, 4, 4)
        plt.plot(time, vll_lines)
        plt.xlabel(r"$t (s)$")
        plt.ylabel(r"$v_\parallel$")
        plt.subplot(2, 4, 5)
        plt.plot(time, (moment_lines - moment_lines[0]) / moment_lines[0])
        plt.xlabel(r"$t (s)$")
        plt.ylabel(r"$(p_\phi-p_{\phi0})/p_{\phi0}$")
        plt.subplot(2, 4, 6)
        plt.plot(time, B_lines)
        plt.xlabel(r"t (s)")
        plt.ylabel(r"|B|")
        # plt.plot(self.time, (self.p_phi - self.p_phi[0]) / self.p_phi[0])
        # plt.xlabel(r"$t (s)$")
        # plt.ylabel(r"$(p_\phi-p_{\phi_initial})/p_{\phi_initial}$")
        # plt.subplot(2, 4, 7)
        # plt.plot(time, B_PHI, label=r"$B_\phi$")
        # plt.plot(time, B_R, label=r"$B_R$")
        # plt.plot(time, B_Z, label=r"$B_Z$")
        # plt.xlabel(r"$t (s)$")
        # plt.legend()
        plt.subplot(2, 4, 7)
        plt.plot(S_lines * np.cos(U_lines), S_lines * np.sin(U_lines))
        plt.xlabel(r"$R$")
        plt.ylabel(r"$Z$")
        plt.subplot(2, 4, 8)
        plt.plot(R_lines, Z_lines)
        plt.xlabel(r"$R$")
        plt.ylabel(r"$Z$")
        plt.tight_layout()
        if show:
            plt.show()

    def plot_orbit_3d(self, show=True):
        """Create 3 dimensional plot of a particle orbit from BEAMS3D"""
        beams3d_file = f"{self.results_folder}/beams3d_{self.equilibrium_name}.h5"
        with h5py.File(beams3d_file, "r") as f:
            R_lines = np.array(f["R_lines"])[0]
            Z_lines = np.array(f["Z_lines"])[0]
            PHI_lines = np.array(f["PHI_lines"])[0]

        X_lines = R_lines * np.cos(PHI_lines)
        Y_lines = R_lines * np.sin(PHI_lines)

        x_surface, y_surface, z_surface = self.field.surface_xyz(
            self.field.nsurfaces - 1
        )
        plot_orbit3d(
            [x_surface, y_surface, z_surface],
            [X_lines, Y_lines, Z_lines],
            distance=6,
            show=show,
        )

        if show:
            plt.show()

    def plot_orbit(self, show=True):
        """Create 2 dimensional plot of a particle orbit from BEAMS3D"""
        beams3d_file = f"{self.results_folder}/beams3d_{self.equilibrium_name}.h5"
        with h5py.File(beams3d_file, "r") as f:
            S_lines = np.array(f["S_lines"])[0]
            U_lines = np.array(f["U_lines"])[0]

        plot_orbit2d(
            x_position=S_lines * np.cos(U_lines),
            y_position=S_lines * np.sin(U_lines),
            show=show,
        )

    def plot_animation(self, distance=7, show=True, save_movie=False):
        """Plot three-dimensional animation of the particle orbit"""
        beams3d_file = f"{self.results_folder}/beams3d_{self.equilibrium_name}.h5"
        with h5py.File(beams3d_file, "r") as f:
            R_lines = np.array(f["R_lines"])[0]
            Z_lines = np.array(f["Z_lines"])[0]
            PHI_lines = np.array(f["PHI_lines"])[0]

        X_lines = R_lines * np.cos(PHI_lines)
        Y_lines = R_lines * np.sin(PHI_lines)

        x_surface, y_surface, z_surface = self.field.surface_xyz(
            self.field.nsurfaces - 1
        )

        plot_animation3d(
            boundary=[x_surface, y_surface, z_surface],
            rpos_cartesian=[X_lines, Y_lines, Z_lines],
            nsamples=self.nsamples,
            distance=distance,
            show=show,
            save_movie=save_movie,
        )
