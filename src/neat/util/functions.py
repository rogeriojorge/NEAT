import os

import numpy as np

from neatpp import gc_solver_qs


# Function called in main.py to run each particle orbit using gyronimo
def orbit(stel, params, B20real):
    """
    Interface function with the C++ executable NEAT. Receives a pyQSC instance
    and outputs the characteristics of the orbit.
    Args:
        stel: Qsc instance of pyQSC
        params (dict): a Python dict() containing the following parameters:
            r0,theta0,phi0,charge,rhom,mass,Lambda,energy,nsamples,Tfinal
        B20real (bool): True if a constant B20real should be used, False otherwise
    """
    # if hasattr(stel.B0, "__len__"):
    #     if stel.order == 'r1':
    #         B20 = [0]*(len(stel.varphi)+1)
    #         B2c = [0]*(len(stel.varphi)+1)
    #         B2s = [0]*(len(stel.varphi)+1)
    #         beta_0  = [0]*(len(stel.varphi)+1)
    #         beta_1c = [0]*(len(stel.varphi)+1)
    #         beta_1s = [0]*(len(stel.varphi)+1)
    #         stel.G2 = 0
    #     else:
    #         if B20real:
    #             B20=np.append(stel.B20,stel.B20[0])
    #         else:
    #             B20=[stel.B20_mean]*(len(stel.varphi)+1)
    #         B2c = [stel.B2c]*(len(stel.varphi)+1)
    #         B2s = [stel.B2s]*(len(stel.varphi)+1)
    #         beta_0  = np.append(stel.beta_0,stel.beta_0[0])
    #         beta_1c = np.append(stel.beta_1c,stel.beta_1c[0])
    #         beta_1s = np.append(stel.beta_1s,stel.beta_1s[0])

    #     # Call Gyronimo
    #     sol = np.array(gc_solver(int(stel.nfp),
    #     stel.G0, stel.G2, stel.I2, stel.iota, stel.iotaN, stel.Bbar,
    #     np.append(stel.varphi,2*np.pi/stel.nfp+stel.varphi[0]),
    #     np.append(stel.B0,stel.B0[0]), np.append(stel.B1c,stel.B1c[0]), np.append(stel.B1s,stel.B1s[0]),
    #     B20, B2c, B2s, beta_0, beta_1c, beta_1s,
    #     params['charge'], params['rhom'], params['mass'], params['Lambda'], params['energy'], params['r0'], params['theta0'], params['phi0'], params['nsamples'], params['Tfinal']))
    # else:
    if stel.order == "r1":
        B20 = 0
        B2c = 0
        beta_1s = 0
        G2 = 0
    else:
        B2c = stel.B2c
        beta_1s = stel.beta_1s
        G2 = stel.G2
        if B20real:
            # B20=np.append(stel.B20,stel.B20[0])
            print("Quasisymmetric NEAT not implemented yet")
            exit()
        else:
            B20 = stel.B20_mean

    # Call Gyronimo
    sol = np.array(
        gc_solver_qs(
            stel.G0,
            G2,
            stel.I2,
            stel.iota,
            stel.iotaN,
            stel.Bbar,
            stel.B0,
            stel.etabar * stel.B0,
            B20,
            B2c,
            beta_1s,
            params["charge"],
            params["rhom"],
            params["mass"],
            params["Lambda"],
            params["energy"],
            params["r0"],
            params["theta0"],
            params["phi0"],
            params["nsamples"],
            params["Tfinal"],
        )
    )

    # Store all output quantities
    time = sol[:, 0]
    r_pos = sol[:, 1]
    theta_pos = sol[:, 2]
    varphi = sol[:, 3]
    from scipy.interpolate import CubicSpline as spline

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

    m_proton = 1.67262192369e-27
    e = 1.602176634e-19
    mu0 = 1.25663706212e-6
    Valfven = 1 / np.sqrt(mu0 * params["rhom"] * m_proton * 1.0e19)

    # Calculate canonical angular momentum p_phi
    m_proton = 1.67262192369e-27
    e = 1.602176634e-19
    mu0 = 1.25663706212e-6
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


def check_log_error(result_array: float) -> float:
    """
    Check the log error for a given array result_array, defined as log(abs((result_array-result_array[0])/result_array))
    """
    array_error = [
        np.log10(np.abs(res - res[0] + 1e-30) / res[0]) for res in result_array
    ]
    max_error = [max(error[3::]) for error in array_error]
    return max_error


def create_BEAMS3D_input(
    name,
    stel,
    rhom,
    mass,
    r0,
    theta0,
    phi0,
    charge,
    Tfinal,
    rParticleVec,
    nsamples,
    zParticleVec,
    phiParticleVec,
    result,
):
    print("Input for BEAMS3D")
    m_proton = 1.67262192369e-27
    e = 1.602176634e-19
    mu0 = 1.25663706212e-6
    Valfven = stel.B0 / np.sqrt(mu0 * rhom * m_proton * 1.0e19)
    Ualfven = 0.5 * m_proton * mass * Valfven * Valfven
    energySI = result[0][4][0] * Ualfven
    Bfield0 = result[0][15][0]
    rStart, zStart, phiStart = stel.to_RZ(r0, theta0, phi0)
    print("  R_START_IN =", rStart)
    print("  Z_START_IN =", zStart)
    print("  PHI_START_IN =", phiStart)
    print("  CHARGE_IN =", e)
    print("  MASS_IN =", mass * m_proton)
    print("  ZATOM_IN =", charge)
    print("  T_END_IN =", Tfinal * stel.rc[0] / Valfven)
    print("  NPOINC =", nsamples)
    print("  VLL_START_IN =", Valfven * result[0][14][0])
    print("  MU_START_IN =", energySI / Bfield0)
    print("")
    os.chdir("results/" + name)
    np.savetxt("rParticleVec.txt", rParticleVec)
    np.savetxt("zParticleVec.txt", zParticleVec)
    np.savetxt("phiParticleVec.txt", phiParticleVec)
    np.savetxt("vparallel.txt", np.array(result[0][14]))
    os.chdir("..")
    os.chdir("..")
