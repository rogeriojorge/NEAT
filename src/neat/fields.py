""" Fields module of NEAT

This script defines the fields being
used by neat (near-axis or VMEC) and
defines what are the gyronimo-based
functions from C++ that are called
for each field. Each classo also contains
the necessary SIMSOPT wrappers for optimization.

"""

import os

import numpy as np
from nptyping import Integer
from qic import Qic
from qsc import Qsc
from scipy.io import netcdf
from simsopt._core.optimizable import Optimizable

from neatpp import (
    gc_solver,
    gc_solver_ensemble,
    gc_solver_qs,
    gc_solver_qs_ensemble,
    gc_solver_qs_partial,
    gc_solver_qs_partial_ensemble,
    vmectrace,
)


class Stellna(Qic, Optimizable):
    """Stellna class

    This class initializes a pyQIC field in the same
    way as a Qic function is used. It also contains
    the necessary SIMSOPT wrappers for optimization.

    """

    def __init__(self, *args, **kwargs) -> None:
        Qic.__init__(self, *args, **kwargs)
        Optimizable.__init__(
            self,
            x0=Qic.get_dofs(self),
            external_dof_setter=Qic.set_dofs,
            names=self.names,
        )

        self.near_axis = True

        assert hasattr(
            self.B0, "__len__"
        ), "The Stellna field requires a magnetic field with B0 a function of phi"

        self.constant_b20 = (
            False  # This variable may be changed later if B20 should be constant
        )

        if self.order == "r1":
            self.B20 = [0] * (len(self.varphi))
            self.B2c_array = [0] * (len(self.varphi))
            self.B2s_array = [0] * (len(self.varphi))
            self.beta_0 = [0] * (len(self.varphi))
            self.beta_1c = [0] * (len(self.varphi))
            self.beta_1s = [0] * (len(self.varphi))
            self.G2 = 0

    def gyronimo_parameters(self):
        """Return list of parameters to feed gyronimo-based functions"""
        if self.order == "r1":
            B20 = np.append(self.B20, self.B20[0])
        else:
            if self.constant_b20:
                B20 = [self.B20_mean] * (len(self.varphi) + 1)
            else:
                B20 = np.append(self.B20, self.B20[0])

        B2c = np.append(self.B2c_array, self.B2c_array[0])
        B2s = np.append(self.B2s_array, self.B2s_array[0])
        beta_0 = np.append(self.beta_0, self.beta_0[0])
        beta_1c = np.append(self.beta_1c, self.beta_1c[0])
        beta_1s = np.append(self.beta_1s, self.beta_1s[0])

        return (
            int(self.nfp),
            self.G0,
            self.G2,
            self.I2,
            self.iota,
            self.iotaN,
            self.Bbar,
            np.append(self.varphi, 2 * np.pi / self.nfp + self.varphi[0]),
            np.append(self.B0, self.B0[0]),
            np.append(self.B1c, self.B1c[0]),
            np.append(self.B1s, self.B1s[0]),
            B20,
            B2c,
            B2s,
            beta_0,
            beta_1c,
            beta_1s,
        )

    def neatpp_solver(self, *args, **kwargs):
        """Specify what gyronimo-based function from neatpp to use as single particle tracer"""
        return gc_solver(*args, *kwargs)

    def neatpp_solver_ensemble(self, *args, **kwargs):
        """Specify what gyronimo-based function from neatpp to use as ensemble particle tracer"""
        return gc_solver_ensemble(*args, *kwargs)

    def get_inv_L_grad_B(self):
        """Wrapper for 1/L_gradB to feed into SIMSOPT"""
        return self.inv_L_grad_B / np.sqrt(self.nphi)

    def get_elongation(self):
        """Wrapper for elongation to feed into SIMSOPT"""
        return self.elongation / np.sqrt(self.nphi)

    def get_B20_mean(self):
        """Wrapper for the mean of B20 to feed into SIMSOPT"""
        return self.B20_mean

    def get_grad_grad_B_inverse_scale_length_vs_varphi(self):
        """Wrapper for 1/L_gradgradB(varphi) to feed into SIMSOPT"""
        return self.grad_grad_B_inverse_scale_length_vs_varphi / np.sqrt(self.nphi)


class StellnaQS(Qsc, Optimizable):
    """Stellna_QS class

    This class initializes a pyQSC field in the same
    way as a Qsc function is used. It also contains
    the necessary SIMSOPT wrappers for optimization.

    """

    def __init__(self, *args, **kwargs) -> None:
        Qsc.__init__(self, *args, **kwargs)
        Optimizable.__init__(
            self,
            x0=Qsc.get_dofs(self),
            external_dof_setter=Qsc.set_dofs,
            names=self.names,
        )

        self.near_axis = True

        assert not hasattr(
            self.B0, "__len__"
        ), "The StellnaQS field requires a magnetic field with B0 a scalar constant"

        self.B1c = self.etabar * self.B0
        self.B1s = 0

        if self.order == "r1":
            self.B20 = 0
            self.B20_mean = 0
            self.B2c = 0
            self.beta_1s = 0
            self.G2 = 0

        # This variable may be changed later before calling gyronimo_parameters
        self.constant_b20 = True

        # The B20 that is outputted to gyronimo is defined here
        # but is overriden when the gyronimo_parameters function
        # is called
        if self.constant_b20:
            self.B20_gyronimo = self.B20_mean
        else:
            self.B20_gyronimo = np.append(
                self.B20, self.B20[0]  # pylint: disable=unsubscriptable-object
            )

        self.B2c_array = self.B2c
        self.B2s_array = 0

    def gyronimo_parameters(self):
        """Return list of parameters to feed gyronimo-based functions"""
        if self.constant_b20:
            self.B20_gyronimo = self.B20_mean
        else:
            self.B20_gyronimo = np.append(
                self.B20, self.B20[0]  # pylint: disable=unsubscriptable-object
            )
        return (
            self.G0,
            self.G2,
            self.I2,
            self.nfp,
            self.iota,
            self.iotaN,
            np.append(self.varphi, 2 * np.pi / self.nfp + self.varphi[0]),
            self.B0,
            self.B1c,
            self.B20_gyronimo,
            self.B2c,
            self.beta_1s,
        )

    def neatpp_solver(self, *args, **kwargs):
        """Specify what gyronimo-based function from neatpp to use as single particle tracer"""
        if self.constant_b20:
            return gc_solver_qs(*args, *kwargs)
        return gc_solver_qs_partial(*args, *kwargs)

    def neatpp_solver_ensemble(self, *args, **kwargs):
        """Specify what gyronimo-based function from neatpp to use as ensemble particle tracer"""
        if self.constant_b20:
            return gc_solver_qs_ensemble(*args, *kwargs)
        return gc_solver_qs_partial_ensemble(*args, *kwargs)

    def get_inv_L_grad_B(self):
        """Wrapper for 1/L_gradB to feed into SIMSOPT"""
        return self.inv_L_grad_B / np.sqrt(self.nphi)

    def get_elongation(self):
        """Wrapper for elongation to feed into SIMSOPT"""
        return self.elongation / np.sqrt(self.nphi)

    def get_B20_mean(self):
        """Wrapper for the mean of B20 to feed into SIMSOPT"""
        return self.B20_mean

    def get_grad_grad_B_inverse_scale_length_vs_varphi(self):
        """Wrapper for 1/L_gradgradB(varphi) to feed into SIMSOPT"""
        return self.grad_grad_B_inverse_scale_length_vs_varphi / np.sqrt(self.nphi)


class Vmec:  # pylint: disable=R0902
    """VMEC class

    This class initializes a VMEC field to be
    ready to be used in the gyronimo-based
    particle tracer.

    """

    def __init__(self, wout_filename: str) -> None:
        self.near_axis = False
        self.wout_filename = os.path.abspath(wout_filename)
        self.equilibrium_name = os.path.basename(wout_filename)[5:-3]

        net_file = netcdf.netcdf_file(wout_filename, "r", mmap=False)
        self.nsurfaces = net_file.variables["ns"][()]
        self.nfp = net_file.variables["nfp"][()]
        self.xn = net_file.variables["xn"][()]  # pylint: disable=C0103
        self.xm = net_file.variables["xm"][()]  # pylint: disable=C0103
        self.xn_nyq = net_file.variables["xn_nyq"][()]
        self.xm_nyq = net_file.variables["xm_nyq"][()]
        self.b0 = net_file.variables["b0"][()]  # pylint: disable=C0103
        self.volavgB = net_file.variables["volavgB"][()]  # pylint: disable=C0103
        self.nmodes = len(self.xn)
        self.raxis_cc = net_file.variables["raxis_cc"][()]
        self.zaxis_cs = net_file.variables["zaxis_cs"][()]
        self.rmnc = net_file.variables["rmnc"][()]
        self.zmns = net_file.variables["zmns"][()]
        self.bmnc = net_file.variables["bmnc"][()]
        self.lasym = net_file.variables["lasym__logical__"][()]
        if self.lasym == 1:
            self.rmns = net_file.variables["rmns"][()]
            self.zmnc = net_file.variables["zmnc"][()]
            self.bmns = net_file.variables["bmns"][()]
            self.zaxis_cc = net_file.variables["zaxis_cc"][()]
            self.raxis_cs = net_file.variables["raxis_cs"][()]
        else:
            self.rmns = 0 * self.rmnc
            self.zmnc = 0 * self.rmnc
            self.bmns = 0 * self.bmnc
            self.zaxis_cc = 0 * self.zaxis_cs
            self.raxis_cs = 0 * self.raxis_cc
        net_file.close()

    def magnetic_axis_rz(self, phi):
        """Return the (R,Z) components of the magnetic axis for a given phi"""
        r_axis = np.sum(
            [raxis_cc * np.cos(phi * i) for i, raxis_cc in enumerate(self.raxis_cc)]
        ) + np.sum(
            [raxis_cs * np.sin(phi * i) for i, raxis_cs in enumerate(self.raxis_cs)]
        )
        z_axis = np.sum(
            [zaxis_cc * np.cos(phi * i) for i, zaxis_cc in enumerate(self.zaxis_cc)]
        ) + np.sum(
            [zaxis_cs * np.sin(phi * i) for i, zaxis_cs in enumerate(self.zaxis_cs)]
        )
        return r_axis, z_axis

    def surface_rz(self, iradius: Integer, ntheta: Integer = 50, nzeta: Integer = 200):
        """Return the (R,Z) components of a flux surface with the index iradius"""
        r_coordinate = np.zeros((ntheta, nzeta))
        z_coordinate = np.zeros((ntheta, nzeta))
        zeta_2d, theta_2d = np.meshgrid(
            np.linspace(0, 2 * np.pi, num=nzeta), np.linspace(0, 2 * np.pi, num=ntheta)
        )
        for imode in range(self.nmodes):
            angle = self.xm[imode] * theta_2d - self.xn[imode] * zeta_2d
            r_coordinate = (
                r_coordinate
                + self.rmnc[iradius, imode] * np.cos(angle)
                + self.rmns[iradius, imode] * np.sin(angle)
            )
            z_coordinate = (
                z_coordinate
                + self.zmns[iradius, imode] * np.sin(angle)
                + self.zmnc[iradius, imode] * np.cos(angle)
            )
        return r_coordinate, z_coordinate

    def surface_xyz(self, iradius: Integer, ntheta: Integer = 50, nzeta: Integer = 200):
        """Return the (x,y,z) components of a flux surface with the index iradius"""
        r_coordinate, z_coordinate = self.surface_rz(iradius, ntheta, nzeta)
        zeta_2d, _ = np.meshgrid(
            np.linspace(0, 2 * np.pi, num=nzeta), np.linspace(0, 2 * np.pi, num=ntheta)
        )
        x_coordinate = r_coordinate * np.cos(zeta_2d)
        y_coordinate = r_coordinate * np.sin(zeta_2d)
        return [x_coordinate, y_coordinate, z_coordinate]

    def b_field_strength(
        self, iradius: Integer, ntheta: Integer = 50, nzeta: Integer = 200
    ):
        """Return the modulus of the magnetic field at a given surface"""
        zeta_2d, theta_2d = np.meshgrid(
            np.linspace(0, 2 * np.pi, num=nzeta), np.linspace(0, 2 * np.pi, num=ntheta)
        )
        for imode, xn_nyq_i in enumerate(self.xn_nyq):
            angle = self.xm_nyq[imode] * theta_2d - xn_nyq_i * zeta_2d
            b_field = (
                b_field
                + self.bmnc[iradius, imode] * np.cos(angle)
                + self.bmns[iradius, imode] * np.sin(angle)
            )
        return b_field

    def gyronimo_parameters(self):
        """Return list of parameters to feed gyronimo-based functions"""
        return [self.wout_filename]

    def neatpp_solver(self, *args, **kwargs):
        """Specify what gyronimo-based function from neatpp to use as single particle tracer"""
        return vmectrace(*args, *kwargs)

    def neatpp_solver_ensemble(self, *args, **kwargs):
        """Specify what gyronimo-based function from neatpp to use as ensemble particle tracer"""
        raise NotImplementedError
