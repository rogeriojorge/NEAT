import logging
import os
import unittest
from unittest.mock import Mock, patch

import numpy as np
from numpy.testing import assert_almost_equal

from neat.fields import Simple, Stellna, StellnaQS, Vmec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NEATtests(unittest.TestCase):
    def test_StellnaQS(self):
        """
        Test that we can obtain qs fields from pyQSC
        using several methods
        """
        assert_almost_equal(
            StellnaQS.from_paper(1).iota, -0.4204733518104154, decimal=10
        )
        assert_almost_equal(
            StellnaQS(
                rc=[1, 0.155, 0.0102],
                zs=[0, 0.154, 0.0111],
                nfp=2,
                etabar=0.64,
                order="r3",
                B2c=-0.00322,
            ).iota,
            -0.4204733518104154,
            decimal=10,
        )
        assert_almost_equal(
            StellnaQS.from_paper("r1 section 5.3").iota, 0.3111813731231253, decimal=10
        )
        assert_almost_equal(
            StellnaQS(
                rc=[1, 0.042],
                zs=[0, -0.042],
                zc=[0, -0.025],
                nfp=3,
                etabar=-1.1,
                sigma0=-0.6,
            ).iota,
            0.3111813731231253,
            decimal=10,
        )

    def test_Stellna(self):
        """
        Test that we can obtain qi fields from pyQIC
        using several methods
        """
        assert_almost_equal(
            Stellna.from_paper("QI").iota, 0.7166463779543341, decimal=10
        )

        assert_almost_equal(
            Stellna(
                rc=[1, 0.155, 0.0102],
                zs=[0, 0.154, 0.0111],
                nfp=2,
                etabar=0.1,
                order="r3",
                B2c=-0.01,
                nphi=251,
            ).iota,
            -0.018692578813516082,
            decimal=10,
        )
        self.assertAlmostEqual(
            StellnaQS.from_paper(1).gyronimo_parameters()[0], 1.0470998216534495
        )
        self.assertAlmostEqual(Stellna.from_paper(1).gyronimo_parameters()[0], 2)

    def setUp(self):
        self.wout_filename = os.path.join(
            os.path.dirname(__file__), "inputs", "wout_ARIESCS.nc"
        )
        self.simple_object = Simple(
            wout_filename=self.wout_filename,
            B_scale=1.0,
            Aminor_scale=1.0,
            multharm=3,
        )
        self.vmec = Vmec(self.wout_filename)
        self.nfp = [2]
        self.G0 = [0.5]
        self.G2 = [0.3]
        self.I2 = [0.2]
        self.iota = [0.1]
        self.iotaN = [0.15]
        self.Bbar = [1.0]
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
            int(self.nfp[0]),
            self.G0[0],
            self.G2[0],
            self.I2[0],
            self.iota[0],
            self.iotaN[0],
            self.Bbar[0],
            np.append(self.varphi, 2 * np.pi / self.nfp[0] + self.varphi[0]),
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

    def test_simple_single_particle_tracer(self):
        # Set up input parameters for the single particle tracer
        Tracy = self.simple_object.tracy
        Rmajor = self.simple_object.Rmajor
        Simple = self.simple_object.simple
        charge = 1.0
        mass = 1.0
        Lambda = 0.5
        vpp_sign = 1.0
        energy = 1.0
        r_initial = 1.0
        theta_initial = 0.0
        phi_initial = 0.0
        nsamples = 100
        tfinal = 1.0

        result = self.simple_object.simple_single_particle_tracer(
            Tracy,
            Rmajor,
            Simple,
            charge,
            mass,
            Lambda,
            vpp_sign,
            energy,
            r_initial,
            theta_initial,
            phi_initial,
            nsamples,
            tfinal,
        )

        expected_shape = (15,)
        self.assertEqual(result[0].shape, expected_shape)

    def test_simple_ensemble_particle_tracer(self):
        # Set up input parameters for the ensemble particle tracer

        Tracy = self.simple_object.tracy
        Rmajor = self.simple_object.Rmajor
        Simple = self.simple_object.simple

        nlambda_trapped = 10
        nlambda_passing = 10
        r_initial = 0.1
        r_max = 0.9
        ntheta = 20
        nphi = 20
        nthreads = 4
        vparallel_over_v_min = -0.3
        vparallel_over_v_max = 0.3
        npoiper = 100
        npoiper2 = 100
        nper = 1000
        nsamples = 2000
        tfinal = 0.001
        r_initial = 0.12
        energy = 3.52e6
        charge = 2
        mass = 4

        result = self.simple_object.simple_ensemble_particle_tracer(
            Tracy,
            Rmajor,
            Simple,
            charge,
            mass,
            energy,
            nlambda_trapped,
            nlambda_passing,
            r_initial,
            r_max,
            ntheta,
            nphi,
            nsamples,
            tfinal,
            nthreads,
            vparallel_over_v_min,
            vparallel_over_v_max,
            npoiper,
            npoiper2,
            nper,
        )

        # Perform assertions on the result

        # Example assertion: check if time array has the correct shape
        expected_shape = (nsamples,)
        self.assertEqual(result[0].shape, expected_shape)

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.vmec.near_axis, False)
        self.assertEqual(self.vmec.wout_filename, self.wout_filename)

    def test_gyronimo_parameters(self):
        expected_parameters = [self.wout_filename]
        self.assertEqual(self.vmec.gyronimo_parameters(), expected_parameters)


if __name__ == "__main__":
    unittest.main()
