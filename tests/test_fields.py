import logging
import unittest

from numpy.testing import assert_almost_equal

from neat.fields import Stellna, StellnaQS

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
