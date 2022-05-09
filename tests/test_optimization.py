import unittest

from neat.fields import stellna_qs
from neat.objectives import optimize_loss_fraction_skeleton
from neat.tracing import charged_particle_ensemble


class NEATtests(unittest.TestCase):
    def test_optimization(self):
        r_max = 0.1
        r_initial = 0.09
        energy = 1e4

        g_field = stellna_qs.from_paper(4, nphi=131)
        g_particle = charged_particle_ensemble(r0=r_initial, energy=energy)
        optimizer = optimize_loss_fraction_skeleton(g_field, g_particle, r_max)
        initial_loss_fraction = optimizer.residual.J()
        optimizer.run(nIterations=4)
        final_loss_fraction = optimizer.residual.J()
        print(" Initial loss fraction: ", initial_loss_fraction)
        print(" Final loss fraction: ", final_loss_fraction)
        assert final_loss_fraction <= initial_loss_fraction
