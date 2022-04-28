# def create_BEAMS3D_input(
#     name,
#     stel,
#     rhom,
#     mass,
#     r0,
#     theta0,
#     phi0,
#     charge,
#     Tfinal,
#     rParticleVec,
#     nsamples,
#     zParticleVec,
#     phiParticleVec,
#     result,
# ):
#     print("Input for BEAMS3D")
#     m_proton = 1.67262192369e-27
#     e = 1.602176634e-19
#     mu0 = 1.25663706212e-6
#     Valfven = stel.B0 / np.sqrt(mu0 * rhom * m_proton * 1.0e19)
#     Ualfven = 0.5 * m_proton * mass * Valfven * Valfven
#     energySI = result[0][4][0] * Ualfven
#     Bfield0 = result[0][15][0]
#     rStart, zStart, phiStart = stel.to_RZ(r0, theta0, phi0)
#     print("  R_START_IN =", rStart)
#     print("  Z_START_IN =", zStart)
#     print("  PHI_START_IN =", phiStart)
#     print("  CHARGE_IN =", e)
#     print("  MASS_IN =", mass * m_proton)
#     print("  ZATOM_IN =", charge)
#     print("  T_END_IN =", Tfinal * stel.rc[0] / Valfven)
#     print("  NPOINC =", nsamples)
#     print("  VLL_START_IN =", Valfven * result[0][14][0])
#     print("  MU_START_IN =", energySI / Bfield0)
#     print("")
#     os.chdir("results/" + name)
#     np.savetxt("rParticleVec.txt", rParticleVec)
#     np.savetxt("zParticleVec.txt", zParticleVec)
#     np.savetxt("phiParticleVec.txt", phiParticleVec)
#     np.savetxt("vparallel.txt", np.array(result[0][14]))
#     os.chdir("..")
#     os.chdir("..")
