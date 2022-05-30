import numpy as np
from qic import Qic
from qsc import Qsc
from simsopt._core.optimizable import Optimizable

from neatpp import gc_solver_qs  # , gc_solver_qs_partial
from neatpp import gc_solver, gc_solver_qs_ensemble


class stellna(Qic, Optimizable):
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
        ), f"The stellna field requires a non-quasisymmetric magnetic field with B0 a function of phi"

        self.B20_constant = (
            False  # This variable may be changed later if B20 should be constant
        )

    def gyronimo_parameters(self):
        if self.order == "r1":
            B20 = [0] * (len(self.varphi) + 1)
            B2c = [0] * (len(self.varphi) + 1)
            B2s = [0] * (len(self.varphi) + 1)
            beta_0 = [0] * (len(self.varphi) + 1)
            beta_1c = [0] * (len(self.varphi) + 1)
            beta_1s = [0] * (len(self.varphi) + 1)
            self.G2 = 0
        else:
            if self.B20_constant:
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
        return gc_solver(*args, *kwargs)

    # def neatpp_solver_ensemble(self, *args, **kwargs):
    #     return gc_solver_ensemble(*args, *kwargs)

    def get_inv_L_grad_B(self):
        return self.inv_L_grad_B / np.sqrt(self.nphi)

    def get_elongation(self):
        return self.elongation / np.sqrt(self.nphi)

    def get_B20_mean(self):
        return self.B20_mean

    def get_grad_grad_B_inverse_scale_length_vs_varphi(self):
        return self.grad_grad_B_inverse_scale_length_vs_varphi / np.sqrt(self.nphi)


class stellna_qs(Qsc, Optimizable):
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
        ), f"The stellna_qs field requires a quasisymmetric magnetic field with B0 a scalar constant"

        self.B1c = self.etabar * self.B0

        if self.order == "r1":
            self.B20 = 0
            self.B20_mean = 0
            self.B2c = 0
            self.beta_1s = 0
            self.G2 = 0

        self.B20_constant = (
            True  # This variable may be changed later if B20 should be constant
        )

    def gyronimo_parameters(self):
        if self.B20_constant:
            B20 = self.B20_mean
        else:
            B20 = np.append(self.B20, self.B20[0])
        return (
            self.G0,
            self.G2,
            self.I2,
            self.nfp,
            self.iota,
            self.iotaN,
            self.B0,
            self.B1c,
            B20,
            self.B2c,
            self.beta_1s,
        )

    def neatpp_solver(self, *args, **kwargs):
        if self.B20_constant:
            return gc_solver_qs(*args, *kwargs)
        else:
            print("gs_solver_partial not implemented yet")
            # return gc_solver_qs_partial(*args, *kwargs)

    def neatpp_solver_ensemble(self, *args, **kwargs):
        return gc_solver_qs_ensemble(*args, *kwargs)

    def get_inv_L_grad_B(self):
        return self.inv_L_grad_B / np.sqrt(self.nphi)

    def get_elongation(self):
        return self.elongation / np.sqrt(self.nphi)

    def get_B20_mean(self):
        return self.B20_mean

    def get_grad_grad_B_inverse_scale_length_vs_varphi(self):
        return self.grad_grad_B_inverse_scale_length_vs_varphi / np.sqrt(self.nphi)
