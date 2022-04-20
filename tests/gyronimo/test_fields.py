import logging
import unittest
from neat.gyronimo.fields import stellna_qs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NEATtests(unittest.TestCase):
    def test_stellna_qs(self):
        """
        Test that we can obtain qs fields from pyQSC
        using several methods
        """
        assert stellna_qs.from_paper(1).iota == -0.4204733518104154
        assert (
            stellna_qs(
                rc=[1, 0.155, 0.0102],
                zs=[0, 0.154, 0.0111],
                nfp=2,
                etabar=0.64,
                order="r3",
                B2c=-0.00322,
            ).iota
            == -0.4204733518104154
        )
        assert stellna_qs.from_paper("r1 section 5.3").iota == 0.3111813731231253
        assert (
            stellna_qs(
                rc=[1, 0.042],
                zs=[0, -0.042],
                zc=[0, -0.025],
                nfp=3,
                etabar=-1.1,
                sigma0=-0.6,
            ).iota
            == 0.3111813731231253
        )
