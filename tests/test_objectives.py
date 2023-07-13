import unittest

from neat.objectives import EffectiveVelocityResidual, LossFractionResidual
from neat.tracing import ChargedParticleEnsemble

from neat.fields import StellnaQS  # isort:skip


class NEATtests(unittest.TestCase):
    def test_LossFraction(self):
        s_initial = 0.01
        energy = 3.52e6
        charge = 2
        mass = 4

        g_field = StellnaQS.from_paper(1)
        g_particle = ChargedParticleEnsemble(
            r_initial=s_initial,
            energy=energy,
            charge=charge,
            mass=mass,
            nlambda_trapped=3,
            nlambda_passing=2,
            ntheta=3,
            nphi=3,
        )
        self.assertAlmostEqual(
            LossFractionResidual(g_field, g_particle).J(), 0.60171388584596, places=1
        )

    def test_EffectiveVelocityResidual(self):
        s_initial = 0.01
        energy = 3.52e6
        charge = 2
        mass = 4

        g_field = StellnaQS.from_paper(1)
        g_particle = ChargedParticleEnsemble(
            r_initial=s_initial,
            energy=energy,
            charge=charge,
            mass=mass,
            nlambda_trapped=3,
            nlambda_passing=2,
            ntheta=3,
            nphi=3,
        )
        self.assertAlmostEqual(
            EffectiveVelocityResidual(g_field, g_particle).J()[0],
            3.513641844631533e-05,
            places=1,
        )


if __name__ == "__main__":
    unittest.main()
