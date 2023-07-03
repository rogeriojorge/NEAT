import unittest
import numpy as np
sys


class GyronimoParametersTestCase(unittest.TestCase):
    def setUp(self):
        # Configuração inicial para os testes
        self.nfp = 2
        self.G0 = 0.5
        self.G2 = 0.3
        self.I2 = 0.2
        self.iota = 0.1
        self.iotaN = 0.15
        self.Bbar = 1.0
        self.varphi = np.array([0.0, 0.5, 1.0])
        self.B0 = np.array([1.0, 0.8, 0.6])
        self.B1c = np.array([0.2, 0.3, 0.4])
        self.B1s = np.array([0.1, 0.15, 0.2])
        self.B20 = np.array([0.05, 0.06, 0.07])
        self.B2c_array = np.array([0.01, 0.02, 0.03])
        self.B2s_array = np.array([0.02, 0.03, 0.04])
        self.beta_0 = np.array([0.8, 0.9, 1.0])
        self.beta_1c = np.array([0.3, 0.4, 0.5])
        self.beta_1s = np.array([0.2, 0.25, 0.3])
        self.expected_result = (
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
            np.append(self.B20, self.B20[0]),
            np.append(self.B2c_array, self.B2c_array[0]),
            np.append(self.B2s_array, self.B2s_array[0]),
            np.append(self.beta_0, self.beta_0[0]),
            np.append(self.beta_1c, self.beta_1c[0]),
            np.append(self.beta_1s, self.beta_1s[0]),
        )
    def test_gyronimo_parameters(self):
        
        result = Stellna(
            self.nfp,
            self.G0,
            self.G2,
            self.I2,
            self.iota,
            self.iotaN,
            self.Bbar,
            self.varphi,
            self.B0,
            self.B1c,
            self.B1s,
            self.B20,
            self.B2c_array,
            self.B2s_array,
            self.beta_0,
            self.beta_1c,
            self.beta_1s,
        )
        # Verifique se o resultado é igual ao valor esperado
        self.assertEqual(result, self.expected_result)
        # Verifique se os arrays no resultado têm o tamanho correto
        self.assertEqual(len(result[7]), len(self.varphi) + 1)
        self.assertEqual(len(result[8]), len(self.B0) + 1)
        self.assertEqual(len(result[9]), len(self.B1c) + 1)
        self.assertEqual(len(result[10]), len(self.B1s) + 1)
        self.assertEqual(len(result[11]), len(self.B20) + 1)
        self.assertEqual(len(result[12]), len(self.B2c_array) + 1)
        self.assertEqual(len(result[13]), len(self.B2s_array) + 1)
        self.assertEqual(len(result[14]), len(self.beta_0) + 1)
        self.assertEqual(len(result[15]), len(self.beta_1c) + 1)
        self.assertEqual(len(result[16]), len(self.beta_1s) + 1)

if __name__ == '__main__':
    unittest.main()