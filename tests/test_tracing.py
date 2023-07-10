import unittest

import matplotlib.pyplot as plt
import numpy as np

from neat.tracing import (
    ChargedParticle,
    ChargedParticleEnsemble,
    ParticleEnsembleOrbit_Simple,
    ParticleOrbit,
)

from pysimple import orbit_symplectic  # isort:skip


class testtracing(unittest.TestCase):
    def importtest_StellnaQS(self):
        try:
            from neat.fields import StellnaQS

            imported = True
        except ImportError:
            imported = False

        self.assertTrue(imported, "Erro ao importar a clase Stellna")

    def importtest_Stellna(self):
        try:
            from neat.fields import Stellna

            imported = True
        except ImportError:
            imported = False

        self.assertTrue(imported, "Erro ao importar a clase Stellna")

    def test_is_alpha_Particle(self):
        obj = ChargedParticle()

        obj.mass = 4
        obj.charge = 2
        obj.energy = 3.52e6

        result = obj.is_alpha_particle()

        self.assertTrue(result)

    def test_is_alpha_particle_Ensemble(self):
        obj = ChargedParticleEnsemble()

        obj.mass = 4
        obj.charge = 2
        obj.energy = 3.52e6

        result = obj.is_alpha_particle()

        self.assertTrue(result)

    # def test_ParticleOrbit_arrays_initialization(self):
    #     obj=ParticleOrbit()

    #     solution = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12]])

    #     obj.p_phi = np.array([1e-16] * len(obj.time))
    #     obj.rpos_cylindrical = np.array([
    #         solution[:, 12],
    #         solution[:, 14],
    #         solution[:, 13]
    #     ])

    #     excepted_p_phi = np.array([1e-16, 1e-16])
    #     excepted_rpros_cylindrical = np.array([
    #         [4,10],
    #         [6,12],
    #         [5,11]
    #     ])

    #     self.assertTrue(np.array_equal(obj.p_phi, exepted_p_phi))
    #     self.assertTrue(np.array_equal(obj.rpos_cylindrical, excepted_rpros_cylindrical))

    def test_get_vmec_boundary(self):

        from neat.fields import Vmec
        import os

        wout_filename = os.path.join(os.path.dirname(__file__), "inputs", "wout_ARIESCS.nc")

        n_samples = 2000
        tfinal = 0.001
        precision = 7
        r_initial = 0.12
        theta_initial = np.pi / 2
        B0=6
        phi_initial = np.pi
        energy = 3.52e6
        charge = 2
        mass = 4
        Lambda = 0.99
        vpp_sign = -1

        g_field = Vmec(wout_filename=wout_filename)

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
            
        )
        np.testing.assert_allclose(
            g_orbit.total_energy,
            [g_orbit.total_energy[0]] * (n_samples + 1),
            rtol=precision,
        )
        np.testing.assert_allclose(
            g_orbit.p_phi, [g_orbit.p_phi[0]] * (n_samples + 1), rtol=precision
        )


    def test_plot_orbit_contourB(self):
        import os

        from neat.fields import Vmec

        wout_filename = os.path.join(
            os.path.dirname(__file__), "inputs", "wout_ARIESCS.nc"
        )

        r_initial = 0.12
        theta_initial = np.pi / 2
        B0 = 6
        phi_initial = np.pi
        energy = 3.52e6
        charge = 2
        mass = 4
        Lambda = 0.99
        vpp_sign = -1

        g_field = Vmec(wout_filename=wout_filename)

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

        obj = ParticleOrbit(g_particle, g_field)

        ntheta = 100
        nphi = 120
        ncontours = 20
        show = False

        try:
            obj.plot_orbit_contourB(ntheta, nphi, ncontours, show)
        except Exception as e:
            self.fail(f"A chamada para plot_orbit_contourB gerou uma exceção: {e}")

        self.assertTrue(True)

    def test_init(self):
        import os

        from neat.fields import StellnaQS

        from neat.fields import Simple  # isort:skip

        s_initial = 0.4
        energy = 3.52e6
        charge = 2
        mass = 4

        wout_filename = os.path.join(
            os.path.dirname(__file__), "inputs", "wout_ARIESCS.nc"
        )
        B_scale = 1
        Aminor_scale = 1

        g_field = Simple(
            wout_filename=wout_filename, B_scale=B_scale, Aminor_scale=Aminor_scale
        )
        g_particle = ChargedParticleEnsemble(
            r_initial=s_initial,
            energy=energy,
            charge=charge,
            mass=mass,
        )
        
        # field = StellnaQS.from_paper(1, B0=1)
        # nsamples = 800
        tfinal = 0.0001
        # nthreads = 2
        nparticles = 32
        

        obj = ParticleEnsembleOrbit_Simple(
            g_particle,
            g_field,
            tfinal=tfinal,
            nparticles=nparticles,
            
        )

        self.assertEqual(obj.particles, g_particle)
        self.assertEqual(obj.nparticles, nparticles)
        # self.assertEqual(obj.particles.ntheta, nparticles)
        # self.assertEqual(obj.particles.nphi, 1)
        # self.assertEqual(obj.particles.nlambda_passing, 1)
        # self.assertEqual(obj.particles.nlambda_trapped, 1)
        self.assertEqual(obj.field, g_field)
        # self.assertEqual(obj.nsamples, nsamples)
        # self.assertEqual(obj.nthreads, nthreads)
        self.assertEqual(obj.tfinal, tfinal)

        # self.assertEqual(len(obj.gyronimo_parameters), 11)

        # self.assertEqual(len(obj.time), nsamples)
        # self.assertEqual(len(obj.confpart_pass), nsamples)
        # self.assertEqual(len(obj.confpart_trap), nsamples)
        # self.assertEqual(len(obj.trace_time), nsamples)
        # self.assertEqual(len(obj.lost_times_of_particles), nparticles)
        # self.assertEqual(len(obj.perp_inv), nsamples)

        # self.assertEqual(len(obj.condi), nparticles)

        # self.assertEqual(len(obj.loss_fraction_array), nsamples)
        # self.assertEqual(obj.total_particles_lost, obj.loss_fraction_array[-1])

        try:
            obj.plot_loss_fraction(show=False)
        except Exception:
            self.fail(f"A chamada para plot_loss_fraction gerou uma exceção")

        self.assertTrue(plt.fignum_exists(1))
        self.assertTrue(plt.fignum_exists(2))

    # def test_plot_loss_fraction(self):
    #     from neat.fields import Simple  # isort:skip
    #     import os

    #     s_initial = 0.4
    #     energy = 3.52e6
    #     charge = 2
    #     mass = 4

    #     wout_filename = os.path.join(
    #         os.path.dirname(__file__), "inputs", "wout_ARIESCS.nc"
    #     )
    #     B_scale = 1
    #     Aminor_scale = 1

    #     g_field = Simple(
    #         wout_filename=wout_filename, B_scale=B_scale, Aminor_scale=Aminor_scale
    #     )
    #     g_particle = ChargedParticleEnsemble(
    #         r_initial=s_initial,
    #         energy=energy,
    #         charge=charge,
    #         mass=mass,
    #     )

    #     obj = ParticleEnsembleOrbit_Simple(g_particle, g_field)

    #     try:
    #         obj.plot_loss_fraction(show=False)
    #     except Exception:
    #         self.fail(f"A chamada para plot_loss_fraction gerou uma exceção")

    #     self.assertTrue(plt.fignum_exists(1))
    #     self.assertTrue(plt.fignum_exists(2))


if __name__ == "__main__":
    unittest.main()
