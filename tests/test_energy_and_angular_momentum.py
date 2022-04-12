import logging
import unittest

import numpy as np
from qsc import Qsc

from neat.util.functions import check_log_error, orbit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QscTests(unittest.TestCase):
    def test_energy_and_angular_momentum(self):
        """
        Test that the energy is constant for a test case

        """
        # Trace orbit for a particle
        stel = Qsc.from_paper(2, nphi=101, B0=3)
        params = {
            "r0": 0.05,
            "theta0": 0,
            "phi0": 0,
            "charge": 1,
            "rhom": 1,
            "mass": 1,
            "Lambda": 0.2,
            "energy": 4e4,
            "nsamples": 1000,
            "Tfinal": 500,
        }
        result = np.array(orbit(stel, params, 0), dtype=object)

        # Check energy error
        logger.info("Energy error = {}".format(check_log_error([result[4]])))
        assert np.allclose(
            check_log_error([result[4]]), -12.879998571350793, rtol=1e-03
        )

        # Check canonical angular momentum error for each orbit
        logger.info(
            "Canonical angular momentum error = {}".format(check_log_error([result[9]]))
        )
        assert np.allclose(check_log_error([result[9]]), -9.219989815050411, rtol=1e-03)
