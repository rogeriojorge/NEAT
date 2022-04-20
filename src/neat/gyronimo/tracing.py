import logging
import numpy as np
from .fields import stellna_qs
from neatpp import gc_solver_qs
from ..util.constants import PROTON_MASS, ELEMENTARY_CHARGE, MU_0
from scipy.interpolate import CubicSpline as spline

logger = logging.getLogger(__name__)

class charged_particle():
    r"""
    Class that contains the physics information of a
    given charged particle, as well as its position
    and velocity
    """
    def __init__(self, charge=1, rhom=1, mass=1, Lambda=0.4,
                 energy=4e4, r0=0.05, theta0=0, phi0=0) -> None:
        self.charge = charge
        self.rhom = rhom
        self.mass = mass
        self.energy = energy
        self.Lambda = Lambda
        self.r0 = r0
        self.theta0 = theta0
        self.phi0 = phi0

    def gyronimo_parameters(self):
        return self.charge, self.rhom, self.mass,\
               self.Lambda, self.energy, self.r0,\
               self.theta0, self.phi0

class particle_orbit():
    r"""
    Interface function with the C++ executable NEAT. Receives a pyQSC instance
    and outputs the characteristics of the orbit.
    Args:
        stel: Qsc instance of pyQSC
        params (dict): a Python dict() containing the following parameters:
            r0,theta0,phi0,charge,rhom,mass,Lambda,energy,nsamples,Tfinal
        B20real (bool): True if a constant B20real should be used, False otherwise
    """
    def __init__(self, particle, field, nsamples=500, Tfinal=1000) -> None:
        solution = np.array(
            gc_solver_qs(
                *field.gyronimo_parameters(),
                *particle.gyronimo_parameters(),
                nsamples,
                Tfinal
            )
        )
        self.gyronimo_parameters = solution

        self.time = solution[:, 0]
        self.r_pos = solution[:, 1]
        self.theta_pos = solution[:, 2]
        self.varphi = solution[:, 3]
        
        nu = field.varphi - field.phi
        nu_spline_of_varphi = spline(
            np.append(field.varphi, 2 * np.pi / field.nfp),
            np.append(nu, nu[0]),
            bc_type="periodic",
        )

        self.phi_pos = self.varphi - nu_spline_of_varphi(self.varphi)
        self.energy_parallel = solution[:, 4]
        self.energy_perpendicular = solution[:, 5]
        self.total_energy = self.energy_parallel + self.energy_perpendicular
        self.Bfield = solution[:, 6]
        self.v_parallel = solution[:, 7]
        self.rdot = solution[:, 8]
        self.thetadot = solution[:, 9]
        self.phidot = solution[:, 10]
        self.vppdot = solution[:, 11]

        self.p_phi = canonical_angular_momentum(particle, field, self.r_pos, self.v_parallel, self.Bfield)

def canonical_angular_momentum(particle, field, r_pos, v_parallel, Bfield):

    m_proton = PROTON_MASS
    e = ELEMENTARY_CHARGE
    mu0 = MU_0
    Valfven = field.Bbar / np.sqrt(mu0 * particle.rhom * m_proton * 1.0e19)

    p_phi1 = (
        particle.mass
        * m_proton
        * v_parallel
        * Valfven
        * (field.G0 + r_pos**2 * (field.G2 + (field.iota - field.iotaN) * field.I2))
        / Bfield
        / field.Bbar
    )
    
    p_phi2 = particle.charge * e * r_pos**2 * field.Bbar / 2 * field.iotaN
    p_phi = p_phi1 - p_phi2

    return p_phi

class trace_particles():
    r"""
    Use gyronimo to trace particles
    """
    def __init__(self) -> None:
        pass
