import logging
import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from neat.fields import Simple, Stellna, StellnaQS

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

    def setUp(self):
        self.simple_object = Simple(
            wout_filename="/home/rodrigo/NEAT/examples/inputs/wout_ARIESCS.nc",
            B_scale=1.0,
            Aminor_scale=1.0,
            multharm=3,
        )

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


if __name__ == "__main__":
    unittest.main()
