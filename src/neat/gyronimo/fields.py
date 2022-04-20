from qsc import Qsc


class stellna_qs:
    r"""
    Create a quasisymmetric stellarator equilibrium
    using the code pyQSC
    https://github.com/landreman/pyQSC
    """

    def __init__(self, *args, **kwargs) -> None:
        self.stel = Qsc(*args, **kwargs)
        self.add_to_self()

    @classmethod
    def from_paper(cls, name, **kwargs) -> None:
        stel = Qsc.from_paper(name, **kwargs)
        return cls(
            rc=stel.rc,
            zs=stel.zs,
            rs=stel.rs,
            zc=stel.zc,
            nfp=stel.nfp,
            etabar=stel.etabar,
            sigma0=stel.sigma0,
            B0=stel.B0,
            I2=stel.I2,
            sG=stel.sG,
            spsi=stel.spsi,
            nphi=stel.nphi,
            B2s=stel.B2s,
            B2c=stel.B2c,
            p2=stel.p2,
            order=stel.order,
        )

    def add_to_self(self) -> None:
        self.G0 = self.stel.G0
        self.iota = self.stel.iota
        self.iotaN = self.stel.iotaN
        self.Bbar = self.stel.Bbar
        self.I2 = self.stel.I2
        self.B0 = self.stel.B0
        self.etabar = self.stel.etabar
        self.varphi = self.stel.varphi
        self.phi = self.stel.phi
        self.nfp = self.stel.nfp
        if self.stel.order == "r1":
            self.B20 = 0
            self.B2c = 0
            self.beta_1s = 0
            self.G2 = 0
        else:
            self.B2c = self.stel.B2c
            self.beta_1s = self.stel.beta_1s
            self.G2 = self.stel.G2
            self.B20 = self.stel.B20_mean

    def gyronimo_parameters(self):
        return (
            self.G0,
            self.G2,
            self.I2,
            self.iota,
            self.iotaN,
            self.Bbar,
            self.B0,
            self.etabar * self.B0,
            self.B20,
            self.B2c,
            self.beta_1s,
        )
