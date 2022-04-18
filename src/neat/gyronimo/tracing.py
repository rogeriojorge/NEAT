import logging
import numpy as np
from fields import stellna_qs
from neatpp import gc_solver_qs
from util.constants import PROTON_MASS, ELEMENTARY_CHARGE, MU_0
from scipy.interpolate import CubicSpline as spline

logger = logging.getLogger(__name__)

class particle():
    r"""
    Class that contains the physics information of a
    given charged particle, as well as its position
    and velocity
    """
    def __init__(self) -> None:
        self.charge = 1
        self.rhom = 1
        self.mass = 1
        self.Lambda = 0.8
        self.energy = 4e4
        self.r0 = 0.05
        self.theta0 = 0
        self.phi0 = 0

    def params(self):
        return self.charge, self.rhom, self.mass,\
               self.Lambda, self.energy, self.r0,\
               self.theta0, self.phi0

def orbit(stel, params, B20real, nsamples, Tfinal):
    r"""
    Interface function with the C++ executable NEAT. Receives a pyQSC instance
    and outputs the characteristics of the orbit.
    Args:
        stel: Qsc instance of pyQSC
        params (dict): a Python dict() containing the following parameters:
            r0,theta0,phi0,charge,rhom,mass,Lambda,energy,nsamples,Tfinal
        B20real (bool): True if a constant B20real should be used, False otherwise
    """
    gyronimo_field = stellna_qs(stel, B20real)
    gyronimo_particle = particle()
    sol = np.array(
        gc_solver_qs(
            gyronimo_field.params(),
            gyronimo_particle.params(),
            nsamples,
            Tfinal
        )
    )

    time = sol[:, 0]
    r_pos = sol[:, 1]
    theta_pos = sol[:, 2]
    varphi = sol[:, 3]
    
    nu = stel.varphi - stel.phi
    nu_spline_of_varphi = spline(
        np.append(stel.varphi, 2 * np.pi / stel.nfp),
        np.append(nu, nu[0]),
        bc_type="periodic",
    )

    phi_pos = varphi - nu_spline_of_varphi(varphi)
    energy_parallel = sol[:, 4]
    energy_perpendicular = sol[:, 5]
    total_energy = energy_parallel + energy_perpendicular
    Bfield = sol[:, 6]
    v_parallel = sol[:, 7]
    rdot = sol[:, 8]
    thetadot = sol[:, 9]
    phidot = sol[:, 10]
    vppdot = sol[:, 11]

    m_proton = PROTON_MASS
    e = ELEMENTARY_CHARGE
    mu0 = MU_0
    Valfven = stel.Bbar / np.sqrt(mu0 * params["rhom"] * m_proton * 1.0e19)

    p_phi1 = (
        params["mass"]
        * m_proton
        * v_parallel
        * Valfven
        * (stel.G0 + r_pos**2 * (stel.G2 + (stel.iota - stel.iotaN) * stel.I2))
        / Bfield
        / stel.Bbar
    )
    
    p_phi2 = params["charge"] * e * r_pos**2 * stel.Bbar / 2 * stel.iotaN
    p_phi = p_phi1 - p_phi2

    return [
        time,
        r_pos,
        theta_pos,
        phi_pos,
        total_energy,
        params["theta0"],
        params["phi0"],
        params["Lambda"],
        params["energy"],
        p_phi,
        rdot,
        thetadot,
        phidot,
        vppdot,
        v_parallel,
        Bfield,
        varphi,
    ]


class trace_particles():
    r"""
    Use gyronimo to trace particles
    """
    def __init__(self) -> None:
        pass
