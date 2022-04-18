
class stellna_qs():
    def __init__(self, stel, B20real = False) -> None:
        if stel.order == "r1":
            self.B20 = 0
            self.B2c = 0
            self.beta_1s = 0
            self.G2 = 0
        else:
            self.B2c = stel.B2c
            self.beta_1s = stel.beta_1s
            self.G2 = stel.G2
            if B20real:
                # B20=np.append(self.B20,self.B20[0])
                print("Quasisymmetric NEAT not implemented yet")
                exit()
            else:
                self.B20 = stel.B20_mean
        self.stel = stel
    def params(self):
        return self.G0, self.G2, self.I2,\
            self.iota, self.iotaN, self.Bbar,\
            self.B0, self.etabar * self.B0,\
            self.B20, self.B2c, self.beta_1s