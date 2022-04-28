import unittest

from neat.fields import stellna_qs
from neat.objectives import optimize_loss_fraction
from neat.tracing import charged_particle_ensemble


class NEATtests(unittest.TestCase):
    def test_optimization(self):
        r_surface_max = 0.1
        r_initial = 0.08
        energy = 1e3

        g_field = stellna_qs.from_paper(2, nphi=101)
        g_particle = charged_particle_ensemble(r0=r_initial, energy=energy)
        optimizer = optimize_loss_fraction(
            g_field, g_particle, r_surface_max=r_surface_max
        )
        initial_loss_fraction = optimizer.loss_fraction.J()
        optimizer.run(nIterations=4)
        final_loss_fraction = optimizer.loss_fraction.J()
        print(" Initial loss fraction: ", initial_loss_fraction)
        print(" Final loss fraction: ", final_loss_fraction)
        assert final_loss_fraction < initial_loss_fraction
