import glob
import os
import unittest

from neat.fields import StellnaQS
from neat.objectives import OptimizeLossFractionSkeleton
from neat.tracing import ChargedParticleEnsemble


class NEATtests(unittest.TestCase):
    def test_optimization(self):
        r_max = 0.07
        r_initial = 0.03
        energy = 1e4

        g_field = StellnaQS.from_paper(4, B0=4, nphi=131)
        g_particle = ChargedParticleEnsemble(r_initial=r_initial, energy=energy)
        optimizer = OptimizeLossFractionSkeleton(g_field, g_particle, r_max)
        initial_loss_fraction = optimizer.residual.J()
        optimizer.run(n_iterations=4)
        final_loss_fraction = optimizer.residual.J()
        print(" Initial loss fraction: ", initial_loss_fraction)
        print(" Final loss fraction: ", final_loss_fraction)
        assert final_loss_fraction <= initial_loss_fraction

        # Remove output files from simsopt
        for f in glob.glob("residuals_202*"):
            os.remove(f)
        for f in glob.glob("simsopt_202*"):
            os.remove(f)
