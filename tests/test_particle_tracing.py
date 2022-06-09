import logging
import time
import unittest

import numpy as np

from neat.fields import StellnaQS
from neat.tracing import (
    ChargedParticle,
    ChargedParticleEnsemble,
    ParticleEnsembleOrbit,
    ParticleOrbit,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NEATtests(unittest.TestCase):
    def test_orbit_energy_momentum(self):
        """
        Test that an orbit traced in a quasisymmetric stellarator
        conserves energy and angular momentum
        """
        n_samples = 2000
        tfinal = 0.001
        precision = 7
        r_initial = 0.06  # meters
        theta_initial = np.pi / 2  # initial poloidal angle
        phi_initial = np.pi  # initial poloidal angle
        B0 = 2  # Tesla, magnetic field on-axis
        energy = 3.52e6  # electron-volt
        charge = 2  # times charge of proton
        mass = 4  # times mass of proton
        Lambda = 0.99  # = mu * B0 / energy
        vpp_sign = -1  # initial sign of the parallel velocity, +1 or -1
        constant_b20 = (
            True  # truly quasi-symmetric field for angular momentum conservation
        )

        g_field = StellnaQS.from_paper(1, B0=B0)
        g_particle = ChargedParticle(
            r_initial=r_initial,
            theta_initial=theta_initial,
            phi_initial=phi_initial,
            energy=energy,
            Lambda=Lambda,
            charge=charge,
            mass=mass,
            vpp_sign=vpp_sign,
        )
        g_orbit = ParticleOrbit(
            g_particle,
            g_field,
            nsamples=n_samples,
            tfinal=tfinal,
            constant_b20=constant_b20,
        )
        np.testing.assert_allclose(
            g_orbit.total_energy,
            [g_orbit.total_energy[0]] * (n_samples + 1),
            rtol=precision,
        )
        np.testing.assert_allclose(
            g_orbit.p_phi, [g_orbit.p_phi[0]] * (n_samples + 1), rtol=precision
        )

    def test_orbit_ensemble(self):
        """
        Test serialization with OpenMP
        """
        nthreads_array = [1, 2]
        nthreads = 2
        r_max = 0.1

        g_field = StellnaQS.from_paper(4)
        g_particle = ChargedParticleEnsemble()
        total_times = [
            self.orbit_time_nthreads(nthread, g_particle, g_field)
            for nthread in nthreads_array
        ]
        # self.assertTrue(
        #     total_times == sorted(total_times, reverse=True),
        #     "The OpenMP parallelization is not working",
        # )
        g_orbits = ParticleEnsembleOrbit(g_particle, g_field, nthreads=nthreads)
        loss_fraction = g_orbits.loss_fraction(r_max=r_max)
        self.assertTrue(
            all(
                loss_fraction[i] <= loss_fraction[i + 1]
                for i in range(len(loss_fraction) - 1)
            ),
            "Loss Fraction is not monotonically increasing",
        )
        self.assertTrue(
            all(i <= 1 for i in loss_fraction), "Loss fraction is not smaller than 1"
        )
        g_orbits.plot_loss_fraction(show=True)

    def orbit_time_nthreads(self, nthreads, g_particle, g_field):
        start_time = time.time()
        ParticleEnsembleOrbit(g_particle, g_field, nthreads=nthreads)
        total_time = time.time() - start_time
        logger.info(f"  With {nthreads} threads took {total_time}s")
        return total_time

    def test_plotting(self):
        n_samples = 800
        tfinal = 0.00002
        r_initial = 0.05  # meters
        theta_initial = np.pi / 2  # initial poloidal angle
        phi_initial = np.pi  # initial poloidal angle
        B0 = 5  # Tesla, magnetic field on-axis
        energy = 3.52e6  # electron-volt
        charge = 2  # times charge of proton
        mass = 4  # times mass of proton
        Lambda = 0.98  # = mu * B0 / energy
        vpp_sign = -1  # initial sign of the parallel velocity, +1 or -1
        g_field = StellnaQS.from_paper(4, B0=B0)
        g_particle = ChargedParticle(
            r_initial=r_initial,
            theta_initial=theta_initial,
            phi_initial=phi_initial,
            energy=energy,
            Lambda=Lambda,
            charge=charge,
            mass=mass,
            vpp_sign=vpp_sign,
        )
        g_orbit = ParticleOrbit(g_particle, g_field, nsamples=n_samples, tfinal=tfinal)
        g_orbit.plot(show=True)
        g_orbit.plot_orbit(show=True)
        g_orbit.plot_orbit_3d(show=True)
        g_orbit.plot_animation(show=True, save_movie=False)


if __name__ == "__main__":
    unittest.main()
