import logging
import unittest

import numpy as np

from neat.gyronimo.fields import stellna_qs
from neat.gyronimo.tracing import (charged_particle, charged_particle_ensemble,
                                   particle_ensemble_orbit, particle_orbit)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NEATtests(unittest.TestCase):
    def test_orbit_energy_momentum(self):
        """
        Test that an orbit traced in a quasisymmetric stellarator
        conserves energy and angular momentum
        """
        n_samples = 100
        Tfinal = 1000
        precision = 6

        g_field = stellna_qs.from_paper(1)
        g_particle = charged_particle()
        g_orbit = particle_orbit(g_particle, g_field, nsamples=n_samples, Tfinal=Tfinal)
        np.testing.assert_array_almost_equal(
            g_orbit.total_energy,
            [g_orbit.total_energy[0]] * (n_samples + 1),
            decimal=precision,
        )
        np.testing.assert_array_almost_equal(
            g_orbit.p_phi, [g_orbit.p_phi[0]] * (n_samples + 1), decimal=precision
        )

    def test_orbit_ensemble(self):
        """
        Test serialization with OpenMP
        """
        g_field = stellna_qs.from_paper(1)
        g_particle = charged_particle_ensemble()
        g_orbit = particle_ensemble_orbit(g_particle, g_field, nthreads=8)
