""" Fields module of NEAT

This script defines the fields being
used by neat (near-axis or VMEC) and
defines what are the gyronimo-based
functions from C++ that are called
for each field. Each classo also contains
the necessary SIMSOPT wrappers for optimization.

"""

try:
    from pysimple import new_vmec_stuff_mod as stuff  # isort:skip
    from pysimple import simple, simple_main, params  # isort:skip
    from pysimple import orbit_symplectic  # isort:skip
    from pysimple import can_to_vmec, vmec_to_can, vmec_to_cyl

    simple_loaded = True
except ImportError as error:
    simple_loaded = False

import copy
import random

import numpy as np
from qic import Qic
from qsc import Qsc
from scipy.io import netcdf

from neatpp import (
    gc_solver,
    gc_solver_ensemble,
    gc_solver_qs,
    gc_solver_qs_ensemble,
    gc_solver_qs_partial,
    gc_solver_qs_partial_ensemble,
    vmectrace,
    vmecloss,
)

from .constants import ELEMENTARY_CHARGE, PROTON_MASS

try:
    from simsopt._core import Optimizable

    simsopt_loaded = True
except ImportError as error:
    simsopt_loaded = False

    class Optimizable:
        def __init__(self, *args, **kwargs):
            pass


class Stellna(Qic, Optimizable):
    """Stellna class

    This class initializes a pyQIC field in the same
    way as a Qic function is used. It also contains
    the necessary SIMSOPT wrappers for optimization.

    """

    def __init__(self, *args, **kwargs) -> None:
        Qic.__init__(self, *args, **kwargs)
        Optimizable.__init__(
            self,
            x0=Qic.get_dofs(self),
            external_dof_setter=Qic.set_dofs,
            names=self.names,
        )

        self.near_axis = True

        assert hasattr(
            self.B0, "__len__"
        ), "The Stellna field requires a magnetic field with B0 a function of phi"

        self.constant_b20 = (
            False  # This variable may be changed later if B20 should be constant
        )

        if self.order == "r1":
            self.B20 = [0] * (len(self.varphi))
            self.B2c_array = [0] * (len(self.varphi))
            self.B2s_array = [0] * (len(self.varphi))
            self.beta_0 = [0] * (len(self.varphi))
            self.beta_1c = [0] * (len(self.varphi))
            self.beta_1s = [0] * (len(self.varphi))
            self.G2 = 0

    def gyronimo_parameters(self):
        """Return list of parameters to feed gyronimo-based functions"""
        if self.order == "r1":
            B20 = np.append(self.B20, self.B20[0])
        else:
            if self.constant_b20:
                B20 = [self.B20_mean] * (len(self.varphi) + 1)
            else:
                B20 = np.append(self.B20, self.B20[0])

        B2c = np.append(self.B2c_array, self.B2c_array[0])
        B2s = np.append(self.B2s_array, self.B2s_array[0])
        beta_0 = np.append(self.beta_0, self.beta_0[0])
        beta_1c = np.append(self.beta_1c, self.beta_1c[0])
        beta_1s = np.append(self.beta_1s, self.beta_1s[0])

        return (
            int(self.nfp),
            self.G0,
            self.G2,
            self.I2,
            self.iota,
            self.iotaN,
            self.Bbar,
            np.append(self.varphi, 2 * np.pi / self.nfp + self.varphi[0]),
            np.append(self.B0, self.B0[0]),
            np.append(self.B1c, self.B1c[0]),
            np.append(self.B1s, self.B1s[0]),
            B20,
            B2c,
            B2s,
            beta_0,
            beta_1c,
            beta_1s,
        )

    def neatpp_solver(self, *args, **kwargs):
        """Specify what gyronimo-based function from neatpp to use as single particle tracer"""
        return gc_solver(*args, *kwargs)

    def neatpp_solver_ensemble(self, *args, **kwargs):
        """Specify what gyronimo-based function from neatpp to use as ensemble particle tracer"""
        return gc_solver_ensemble(*args, *kwargs)

    def get_inv_L_grad_B(self):
        """Wrapper for 1/L_gradB to feed into SIMSOPT"""
        return self.inv_L_grad_B / np.sqrt(self.nphi)

    def get_elongation(self):
        """Wrapper for elongation to feed into SIMSOPT"""
        return self.elongation / np.sqrt(self.nphi)

    def get_B20_mean(self):
        """Wrapper for the mean of B20 to feed into SIMSOPT"""
        return self.B20_mean

    def get_grad_grad_B_inverse_scale_length_vs_varphi(self):
        """Wrapper for 1/L_gradgradB(varphi) to feed into SIMSOPT"""
        return self.grad_grad_B_inverse_scale_length_vs_varphi / np.sqrt(self.nphi)


class StellnaQS(Qsc, Optimizable):
    """Stellna_QS class

    This class initializes a pyQSC field in the same
    way as a Qsc function is used. It also contains
    the necessary SIMSOPT wrappers for optimization.

    """

    def __init__(self, *args, **kwargs) -> None:
        Qsc.__init__(self, *args, **kwargs)
        Optimizable.__init__(
            self,
            x0=Qsc.get_dofs(self),
            external_dof_setter=Qsc.set_dofs,
            names=self.names,
        )

        self.near_axis = True

        assert not hasattr(
            self.B0, "__len__"
        ), "The StellnaQS field requires a magnetic field with B0 a scalar constant"

        self.B1c = self.etabar * self.B0
        self.B1s = 0

        if self.order == "r1":
            self.B20 = 0
            self.B20_mean = 0
            self.B2c = 0
            self.beta_1s = 0
            self.G2 = 0

        # This variable may be changed later before calling gyronimo_parameters
        self.constant_b20 = True

        # The B20 that is outputted to gyronimo is defined here
        # but is overriden when the gyronimo_parameters function
        # is called
        if self.constant_b20:
            self.B20_gyronimo = self.B20_mean
        else:
            self.B20_gyronimo = np.append(
                self.B20, self.B20[0]  # pylint: disable=unsubscriptable-object
            )

        self.B2c_array = self.B2c
        self.B2s_array = 0

    def gyronimo_parameters(self):
        """Return list of parameters to feed gyronimo-based functions"""
        if self.constant_b20:
            self.B20_gyronimo = self.B20_mean
        else:
            self.B20_gyronimo = np.append(
                self.B20, self.B20[0]  # pylint: disable=unsubscriptable-object
            )
        return (
            self.G0,
            self.G2,
            self.I2,
            self.nfp,
            self.iota,
            self.iotaN,
            np.append(self.varphi, 2 * np.pi / self.nfp + self.varphi[0]),
            self.B0,
            self.B1c,
            self.B20_gyronimo,
            self.B2c,
            self.beta_1s,
        )

    def neatpp_solver(self, *args, **kwargs):
        """Specify what gyronimo-based function from neatpp to use as single particle tracer"""
        if self.constant_b20:
            return gc_solver_qs(*args, *kwargs)
        return gc_solver_qs_partial(*args, *kwargs)

    def neatpp_solver_ensemble(self, *args, **kwargs):
        """Specify what gyronimo-based function from neatpp to use as ensemble particle tracer"""
        if self.constant_b20:
            return gc_solver_qs_ensemble(*args, *kwargs)
        return gc_solver_qs_partial_ensemble(*args, *kwargs)

    def get_inv_L_grad_B(self):
        """Wrapper for 1/L_gradB to feed into SIMSOPT"""
        return self.inv_L_grad_B / np.sqrt(self.nphi)

    def get_elongation(self):
        """Wrapper for elongation to feed into SIMSOPT"""
        return self.elongation / np.sqrt(self.nphi)

    def get_B20_mean(self):
        """Wrapper for the mean of B20 to feed into SIMSOPT"""
        return self.B20_mean

    def get_grad_grad_B_inverse_scale_length_vs_varphi(self):
        """Wrapper for 1/L_gradgradB(varphi) to feed into SIMSOPT"""
        return self.grad_grad_B_inverse_scale_length_vs_varphi / np.sqrt(self.nphi)


if simple_loaded:

    class Simple:
        """SIMPLE class
        This class initializes a SIMPLE-based particle
        tracer that reads vmec input files.
        SIMPLE code: https://github.com/itpplasma/SIMPLE
        """

        def __init__(
            self,
            wout_filename: str,
            B_scale: float = 1,
            Aminor_scale: float = 1,
            multharm: int = 3,
            ns_s: int = 3,
            ns_tp: int = 3,
            nsamples: int = 20000,
            integmode: int = 1,
            npoiper: int = 100,
            npoiper2: int = 128,
            nper: int =1000,
        ) -> None:

            self.near_axis = False
            self.wout_filename = wout_filename
            net_file = netcdf.netcdf_file(self.wout_filename, "r", mmap=False)
            self.nfp = net_file.variables["nfp"][()]
            self.Rmajor = net_file.variables["Rmajor_p"][()]
            net_file.close()
            self.B_scale = B_scale
            self.Aminor_scale = Aminor_scale
            self.multharm = multharm

            from pysimple import new_vmec_stuff_mod as stuff_local
            from pysimple import params as params_local
            from pysimple import simple as simple_local

            self.params = copy.deepcopy(params_local)
            # self.params.ntimstep = nsamples
            self.params.integmode = integmode
            # self.params.npoiper2 = npoiper2
            # self.params.npoiper = npoiper
            # self.params.npoi = npoiper * nper
            # self.params.nper = nper
            self.stuff = copy.deepcopy(stuff_local)
            self.simple = copy.deepcopy(simple_local)
            
            self.tracy = self.params.Tracer()
            self.stuff.vmec_b_scale = self.B_scale
            self.stuff.vmec_rz_scale = self.Aminor_scale
            self.stuff.multharm = self.multharm
            self.stuff.ns_s = ns_s
            self.stuff.ns_tp = ns_tp

            self.simple.init_field(
                self.tracy,
                self.wout_filename,
                self.stuff.ns_s,
                self.stuff.ns_tp,
                self.stuff.multharm,
                self.params.integmode,
            )

        def gyronimo_parameters(self):
            """Return list of parameters to feed gyronimo-based functions"""
            return [
                self.tracy,
                self.Rmajor,
                self.simple,
            ]

        def neatpp_solver(self, *args, **kwargs):
            """Specify what gyronimo-based function from neatpp to use as single particle tracer"""
            return self.simple_single_particle_tracer(*args, *kwargs)

        def neatpp_solver_ensemble(self, *args, **kwargs):
            """Specify what gyronimo-based function from neatpp to use as ensemble particle tracer"""
            return self.simple_ensemble_particle_tracer(*args, *kwargs)

        def simple_single_particle_tracer(
            self,
            Tracy,
            Rmajor,
            Simple: simple,
            charge,
            mass,
            Lambda,
            vpp_sign,
            energy,
            r_initial,
            theta_initial,
            phi_initial,
            nsamples,
            tfinal,
        ):
            """Single particle tracer that uses SIMPLE's fortran (f90wrap+f2py) compiled functions"""

            relative_error = 1e-13
            # npoints=3000
            npoints=256
            # Simple.init_params(Tracy, charge, mass, energy, npoints, 1, relative_error)
            # self.params.nper=npoints
            Simple.init_params(Tracy, charge, mass, energy, npoints,  1, relative_error)
            # self.params.n_e = charge
            # self.params.n_d = mass
            # self.params.trace_time = tfinal
            # self.params.sbeg = r_initial
            # self.params.params_init()

            # self.tracy.dtau= 1 * (2 * np.pi * Rmajor / npoints)
            # self.tracy.dtaumin= 2 * np.pi * Rmajor / npoints
            # self.tracy.v0 = self.params.v0
            # self.tracy.n_e = charge
            # self.tracy.n_d = mass
            # self.tracy.relerr = relative_error
            

            # s, th, ph, v/v_th, v_par/v
            abs_v_parallel_over_v = np.sqrt(1 - Lambda)
            z0_vmec = np.array(
                [
                    r_initial,
                    theta_initial,
                    phi_initial,
                    1.0,
                    vpp_sign * abs_v_parallel_over_v,
                ]
            )
            z0_can = z0_vmec.copy()

            z0_can[1:3] = vmec_to_can(z0_vmec[0], z0_vmec[1], z0_vmec[2])

            self.simple.init_integrator(Tracy, z0_can)

            # nt = nsamples
            dtaumin = 2 * np.pi * Rmajor / npoints
            v_th = np.sqrt(2 * energy * ELEMENTARY_CHARGE / (mass * PROTON_MASS))
            nt = int(tfinal * v_th / dtaumin)
            if nsamples<nt: print(f'Warning: nsamples={nsamples} smaller than nt={nt}')
            time = np.linspace(dtaumin / v_th, nt * dtaumin / v_th, nt)
            # dtaumin (time step of the integrator) = 2*pi*Rmajor/npoiper2
            # actual time step dt = dtaumin/v_th
            # v_th = sqrt(2*Ekin/mass)
            # Rmajor = Rmajor_p from VMEC
            z_integ = np.zeros([nt, 4])  # s, th_c, ph_c, p_phi
            z_vmec = np.zeros([nt, 5])  # s, th, ph, v/v_th, v_par/v
            z_cyl = np.zeros([nt, 3])
            z_integ[0, :] = Tracy.si.z
            z_vmec[0, :] = z0_vmec
            z_cyl[0, :2] = vmec_to_cyl(z_vmec[0, 0], z_vmec[0, 1], z_vmec[0, 2])
            z_cyl[0, 2] = z_vmec[0, 2]

            for kt in range(nt - 1):
                orbit_symplectic.orbit_timestep_sympl(Tracy.si, Tracy.f)
                z_integ[kt + 1, :] = Tracy.si.z
                z_vmec[kt + 1, 0] = z_integ[kt + 1, 0]
                z_vmec[kt + 1, 1:3] = can_to_vmec(
                    z_integ[kt + 1, 0], z_integ[kt + 1, 1], z_integ[kt + 1, 2]
                )
                z_vmec[kt + 1, 3] = np.sqrt(
                    Tracy.f.mu * Tracy.f.bmod + 0.5 * Tracy.f.vpar**2
                )
                z_vmec[kt + 1, 4] = Tracy.f.vpar / (z_vmec[kt + 1, 3] * np.sqrt(2))
                z_cyl[kt + 1, :2] = vmec_to_cyl(
                    z_vmec[kt + 1, 0], z_vmec[kt + 1, 1], z_vmec[kt + 1, 2]
                )
                z_cyl[kt + 1, 2] = z_vmec[kt + 1, 2]
            ## PARALLEL VELOCITY OUTPUTED WITH A MINUS SIGN
            return np.array(
                [
                    time,
                    z_vmec[:, 0],
                    z_vmec[:, 1],
                    z_vmec[:, 2],
                    np.array(
                        [np.sqrt(2 * energy / mass) * (1 - Lambda)] * len(z_vmec[:, 2])
                    ),  # parallel energy
                    np.array(
                        [Lambda * np.sqrt(2 * energy / mass)] * len(z_vmec[:, 2])
                    ),  # perpendicular energy
                    np.array([0] * len(z_vmec[:, 2])),  # magnetic_field_strength,
                    -z_vmec[:, 4]*v_th,  # parallel velocity
                    np.array([0] * len(z_vmec[:, 2])),  # rdot,
                    np.array([0] * len(z_vmec[:, 2])),  # thetadot,
                    np.array([0] * len(z_vmec[:, 2])),  # varphidot,
                    np.array([0] * len(z_vmec[:, 2])),  # vparalleldot,
                    z_cyl[:, 0]/100,
                    z_cyl[:, 2],
                    z_cyl[:, 1]/100,
                ]
            ).T

        def simple_ensemble_particle_tracer(
            self,
            Tracy,
            Rmajor,
            simple,
            charge,
            mass,
            energy,
            nlambda_trapped,
            nlambda_passing,
            r_initial,
            r_max,
            ntheta,
            nphi,
            nsamples,
            nparticles,
            tfinal,
            nthreads,
            notrace_passing,
            npoiper,
            npoiper2,
            nper,
        ):
            """Ensemble particle tracer that uses SIMPLE's fortran (f90wrap+f2py) compiled functions"""
            # nparticles = ntheta * nphi * nlambda_passing * nlambda_trapped
            # Tracy = self.params.Tracer()

            self.params.ntestpart = nparticles
            self.params.trace_time = tfinal
            self.params.contr_pp = -1e10  # Trace all passing particles
            self.params.startmode = (
                1  # automatically select initial particle distribution
            )
            self.params.ntimstep = nsamples
            self.params.sbeg = r_initial
            self.params.npoiper2 = npoiper2
            self.params.npoiper = npoiper
            self.params.nper = nper
            self.params.n_e = charge
            self.params.n_d = mass
            self.params.notrace_passing = notrace_passing

            self.params.params_init()

            from pysimple import simple_main as simple_main_local

            self.simple_main = copy.deepcopy(simple_main_local)
            self.simple_main.run(Tracy)

            self.time = np.linspace(
                self.params.dtau / self.params.v0,
                self.params.trace_time,
                self.params.ntimstep,
            )
            # condi = np.logical_and(params.times_lost > 0, params.times_lost < params.trace_time)

            return_array = copy.deepcopy(
                (
                    self.time,
                    self.params.confpart_pass,
                    self.params.confpart_trap,
                    self.params.trace_time,
                    self.params.times_lost,
                    self.params.perp_inv,
                )
            )

            # self.simple_main.finalize()

            return return_array


class Vmec:
    """VMEC class

    This class initializes a VMEC field to be
    ready to be used in the gyronimo-based
    particle tracer.

    """

    def __init__(self, wout_filename: str, maximum_s=0.95, integrator=2) -> None:
        self.near_axis = False
        self.wout_filename = wout_filename
        net_file = netcdf.netcdf_file(wout_filename, "r", mmap=False)
        self.nfp = net_file.variables["nfp"][()]
        self.maximum_s = maximum_s
        self.integrator=integrator
        net_file.close()

    def gyronimo_parameters(self):
        """Return list of parameters to feed gyronimo-based functions"""
        return [self.wout_filename, self.maximum_s, self.integrator]

    def neatpp_solver(self, *args, **kwargs):
        """Specify what gyronimo-based function from neatpp to use as single particle tracer"""
        return vmectrace(*args, *kwargs)

    def neatpp_solver_ensemble(self, *args, **kwargs):
        """Specify what gyronimo-based function from neatpp to use as ensemble particle tracer"""
        return vmecloss(*args, *kwargs)
