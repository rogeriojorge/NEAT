from qsc import Qsc
from simsopt._core.optimizable import Optimizable


class stellna_qs(Qsc, Optimizable):
    def __init__(self, *args, **kwargs) -> None:
        Qsc.__init__(self, *args, **kwargs)
        Optimizable.__init__(
            self,
            x0=Qsc.get_dofs(self),
            external_dof_setter=Qsc.set_dofs,
            names=self.names,
        )

        if self.order == "r1":
            self.B20 = 0
            self.B20_mean = 0
            self.B2c = 0
            self.beta_1s = 0
            self.G2 = 0

    def gyronimo_parameters(self):
        return (
            self.G0,
            self.G2,
            self.I2,
            self.nfp,
            self.iota,
            self.iotaN,
            self.B0,
            self.etabar * self.B0,
            self.B20_mean,
            self.B2c,
            self.beta_1s,
        )

    def get_inv_L_grad_B(self):
        return self.inv_L_grad_B / self.nphi

    def get_elongation(self):
        return self.elongation / self.nphi

    def get_B20_mean(self):
        return self.B20_mean

    def get_grad_grad_B_inverse_scale_length_vs_varphi(self):
        return self.grad_grad_B_inverse_scale_length_vs_varphi / self.nphi
