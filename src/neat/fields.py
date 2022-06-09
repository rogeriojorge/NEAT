""" Fields module of NEAT

This script defines the fields being
used by neat (near-axis or VMEC) and
defines what are the gyronimo-based
functions from C++ that are called
for each field. Each classo also contains
the necessary SIMSOPT wrappers for optimization.

"""

import numpy as np
from qic import Qic
from qsc import Qsc
from simsopt._core.optimizable import Optimizable

from neatpp import (
    gc_solver,
    gc_solver_ensemble,
    gc_solver_qs,
    gc_solver_qs_ensemble,
    gc_solver_qs_partial,
    gc_solver_qs_partial_ensemble,
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
