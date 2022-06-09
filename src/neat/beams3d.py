#!/usr/bin/env python3

# def create_BEAMS3D_input(
#     name,
#     stel,
#     rhom,
#     mass,
#     r_initial,
#     theta_initial,
#     phi_initial,
#     charge,
#     tfinal,
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
#     magnetic_field_strength0 = result[0][15][0]
#     rStart, zStart, phiStart = stel.to_RZ(r_initial, theta_initial, phi_initial)
#     print("  R_START_IN =", rStart)
#     print("  Z_START_IN =", zStart)
#     print("  PHI_START_IN =", phiStart)
#     print("  CHARGE_IN =", e)
#     print("  MASS_IN =", mass * m_proton)
#     print("  ZATOM_IN =", charge)
#     print("  T_END_IN =", tfinal * stel.rc[0] / Valfven)
#     print("  NPOINC =", nsamples)
#     print("  VLL_START_IN =", Valfven * result[0][14][0])
#     print("  MU_START_IN =", energySI / magnetic_field_strength0)
#     print("")
#     os.chdir("results/" + name)
#     np.savetxt("rParticleVec.txt", rParticleVec)
#     np.savetxt("zParticleVec.txt", zParticleVec)
#     np.savetxt("phiParticleVec.txt", phiParticleVec)
#     np.savetxt("vparallel.txt", np.array(result[0][14]))
#     os.chdir("..")
#     os.chdir("..")


# import os
# import sys

# import matplotlib.pyplot as plt
# import numpy as np
# from constants import (
#     ALPHA_PARTICLE_MASS,
#     ELEMENTARY_CHARGE,
#     FUSION_ALPHA_PARTICLE_ENERGY,
# )
# from scipy.interpolate import RectBivariateSpline, interp1d
# from scipy.io import netcdf


# class beams3D_from_vmec:
#     """
#     This class creates initial conditions that are randomly
#     distributed over the first field period of a surface and over
#     velocity-space. Ntheta and Nphi are the resolution parameters
#     used for the 2D splines (nphi is per 1 field period)
#     """

#     def __init__(self, filename, s0=0.25, nparticles=100, ntheta=60, nphi=100) -> None:
#         # Add variables to self
#         self.filename = filename
#         self.s0 = s0
#         self.nparticles = nparticles
#         self.ntheta = ntheta
#         self.nphi = nphi

#         self.justwout = os.path.basename(self.filename)
#         self.outfile = "particles.{}_volumeJacobian_s{}_n{}".format(
#             self.justwout[5:-3], s0, nparticles
#         )
#         self.beams3dfile = "beams3d_in.{}_volumeJacobian_s{}_n{}".format(
#             self.justwout[5:-3], s0, nparticles
#         )

#         print("filename: ", self.filename)
#         print("outfile: ", self.outfile)
#         print("beams3dfile: ", self.beams3dfile)
#         print("Requested normalized flux: ", self.s0)
#         print("nparticles: ", self.nparticles)

#         # Data for alpha particles:
#         self.e_C = ELEMENTARY_CHARGE  # Charge of the electron, NOT of the alpha!
#         self.m_kg = ALPHA_PARTICLE_MASS  # Mass of the alpha, NOT of a proton!
#         energy_J = FUSION_ALPHA_PARTICLE_ENERGY  # Alpha birth energy, in Joules
#         self.v = np.sqrt(2 * energy_J / self.m_kg)  # Alpha birth speed [meter/second]
#         print("v [m/s]:", self.v)

#         self.read_vmec()
#         self.compute_jacobian()
#         self.compute_initial_conditions()
#         self.create_outfile()
#         self.create_beams3dfile()
#         self.plot_phitheta()

#     def read_vmec(self):

#         f = netcdf.netcdf_file(self.filename, "r", mmap=False)

#         self.nfp = f.variables["nfp"][()]
#         self.ns = f.variables["ns"][()]
#         self.xm = f.variables["xm"][()]
#         self.xn = f.variables["xn"][()]
#         self.xm_nyq = f.variables["xm_nyq"][()]
#         self.xn_nyq = f.variables["xn_nyq"][()]
#         self.rmnc = f.variables["rmnc"][()]
#         self.zmns = f.variables["zmns"][()]
#         self.gmnc = f.variables["gmnc"][()]
#         self.bmnc = f.variables["bmnc"][()]

#         # bmnc and gmnc are on the half grid and use the Nyquist modes.
#         # rmnc and zmns are on the full grid and use the non-Nyquist modes.

#         f.close()

#         self.s_full = np.linspace(0, 1, self.ns)
#         ds = self.s_full[1] - self.s_full[0]
#         self.s_half = self.s_full[1:] - ds / 2
#         print("rmnc.shape: ", self.rmnc.shape)

#     def compute_jacobian(self, interp_method="linear"):
#         ntheta = self.ntheta
#         nphi = self.nphi
#         index = self.ns - 1
#         print("s_full[index]: ", self.s_full[index])
#         old_rmnc = self.rmnc[index, :]
#         # print('rmnc[index,:]: ', old_rmnc)
#         rc = interp1d(self.s_full, self.rmnc, axis=0, kind=interp_method)(self.s0)
#         zs = interp1d(self.s_full, self.zmns, axis=0, kind=interp_method)(self.s0)
#         # print('New rmnc:', self.rmnc)

#         print("Difference between old_rmnc and rc:", np.max(np.abs(old_rmnc - rc)))

#         # If s0 is close to 1, we may need to extrapolate off the end of the half grid.
#         bc = interp1d(
#             self.s_half,
#             self.bmnc[1:, :],
#             axis=0,
#             kind=interp_method,
#             fill_value="extrapolate",
#         )(self.s0)
#         gc = interp1d(
#             self.s_half,
#             self.gmnc[1:, :],
#             axis=0,
#             kind=interp_method,
#             fill_value="extrapolate",
#         )(self.s0)

#         # theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
#         # phi1d = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
#         # dtheta = theta1d[1] - theta1d[0]
#         # dphi = phi1d[1] - phi1d[0]

#         theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=True)
#         phi1d = np.linspace(0, 2 * np.pi / self.nfp, nphi, endpoint=True)

#         dtheta = theta1d[1] - theta1d[0]
#         dphi = phi1d[1] - phi1d[0]

#         phi, theta = np.meshgrid(phi1d, theta1d)
#         modB = np.zeros((ntheta, nphi))
#         sqrtg = np.zeros((ntheta, nphi))
#         r = np.zeros((ntheta, nphi))
#         x = np.zeros((ntheta, nphi))
#         y = np.zeros((ntheta, nphi))
#         z = np.zeros((ntheta, nphi))
#         dxdtheta = np.zeros((ntheta, nphi))
#         dydtheta = np.zeros((ntheta, nphi))
#         dzdtheta = np.zeros((ntheta, nphi))
#         dxdphi = np.zeros((ntheta, nphi))
#         dydphi = np.zeros((ntheta, nphi))
#         dzdphi = np.zeros((ntheta, nphi))
#         sinphi = np.sin(phi)
#         cosphi = np.cos(phi)
#         for imn in range(len(self.xm_nyq)):
#             m = self.xm_nyq[imn]
#             n = self.xn_nyq[imn]
#             angle = m * theta - n * phi
#             cosangle = np.cos(angle)
#             modB += bc[imn] * cosangle
#             sqrtg += gc[imn] * cosangle

#         for imn in range(len(self.xm)):
#             m = self.xm[imn]
#             n = self.xn[imn]
#             angle = m * theta - n * phi
#             sinangle = np.sin(angle)
#             cosangle = np.cos(angle)
#             rmnc = rc[imn]
#             zmns = zs[imn]
#             r += rmnc * cosangle
#             x += rmnc * cosangle * cosphi
#             y += rmnc * cosangle * sinphi
#             z += zmns * sinangle

#             dxdtheta += rmnc * (-m * sinangle) * cosphi
#             dydtheta += rmnc * (-m * sinangle) * sinphi
#             dzdtheta += zmns * m * cosangle

#             dxdphi += rmnc * (n * sinangle * cosphi + cosangle * (-sinphi))
#             dydphi += rmnc * (n * sinangle * sinphi + cosangle * cosphi)
#             dzdphi += zmns * (-n * cosangle)
#             if False:
#                 # Eventually we might include non-stellarator-symmetric cases:
#                 rmns = self.get_rs(m, n_without_nfp)
#                 zmnc = self.get_zc(m, n_without_nfp)
#                 r += rmns * sinangle
#                 x += rmns * sinangle * cosphi
#                 y += rmns * sinangle * sinphi
#                 z += zmnc * cosangle

#                 dxdtheta += rmns * (m * cosangle) * cosphi
#                 dydtheta += rmns * (m * cosangle) * sinphi
#                 dzdtheta += zmnc * (-m * sinangle)

#                 dxdphi += rmns * (-n * cosangle * cosphi + sinangle * (-sinphi))
#                 dydphi += rmns * (-n * cosangle * sinphi + sinangle * cosphi)
#                 dzdphi += zmnc * (n * sinangle)

#         normalx = dydphi * dzdtheta - dzdphi * dydtheta
#         normaly = dzdphi * dxdtheta - dxdphi * dzdtheta
#         normalz = dxdphi * dydtheta - dydphi * dxdtheta
#         norm_normal = np.sqrt(normalx * normalx + normaly * normaly + normalz * normalz)
#         area = (
#             np.sum(norm_normal[:-1, :-1]) * dtheta * dphi * self.nfp
#         )  # Note we must drop repeated grid points in norm_normal
#         print("Computed area of the surface: ", area)

#         # Use |sqrtg| instead of sqrtg from here onward:
#         sqrtg = np.abs(sqrtg)
#         print("min(sqrtg)={}, max(sqrtg)={}".format(np.min(sqrtg), np.max(sqrtg)))
#         # sqrtg = sqrtg ** 4
#         # print('After ** 4, min(sqrtg)={}, max(sqrtg)={}'.format(np.min(sqrtg), np.max(sqrtg)))

#         self.phi = phi
#         self.theta = theta
#         self.theta1d = theta1d
#         self.phi1d = phi1d
#         self.r = r
#         self.z = z
#         self.modB = modB
#         self.sqrtg = sqrtg
#         self.norm_normal = norm_normal

#     def compute_initial_conditions(self):
#         # Initialize 2D splines
#         r_spl = RectBivariateSpline(self.theta1d, self.phi1d, self.r)
#         z_spl = RectBivariateSpline(self.theta1d, self.phi1d, self.z)
#         modB_spl = RectBivariateSpline(self.theta1d, self.phi1d, self.modB)
#         sqrtg_spl = RectBivariateSpline(self.theta1d, self.phi1d, self.sqrtg)
#         norm_normal_spl = RectBivariateSpline(
#             self.theta1d, self.phi1d, self.norm_normal
#         )

#         thetas = np.zeros(self.nparticles)
#         phis = np.zeros(self.nparticles)
#         # rs = np.zeros(nparticles)
#         # zs = np.zeros(nparticles)
#         # vlls = np.zeros(nparticles)
#         # mus = np.zeros(nparticles)

#         # rng = np.random.default_rng()
#         rng = np.random
#         max_norm_normal = np.max(self.norm_normal)
#         max_sqrtg = np.max(self.sqrtg)
#         for j in range(self.nparticles):
#             # print('j={}: '.format(j), end='')
#             while True:
#                 print(".", end="")
#                 theta_initial = rng.random() * 2 * np.pi
#                 phi_initial = rng.random() * 2 * np.pi / self.nfp
#                 # f = rng.random() * max_norm_normal
#                 # if f <= norm_normal_spl(theta_initial, phi_initial):
#                 #    break
#                 f = rng.random() * max_sqrtg
#                 if f <= sqrtg_spl(theta_initial, phi_initial):
#                     break
#             thetas[j] = theta_initial
#             phis[j] = phi_initial
#             # print()

#         self.rs = r_spl.ev(thetas, phis)
#         self.zs = z_spl.ev(thetas, phis)
#         self.thetas = thetas
#         self.phis = phis

#         # Random numbers on [-1, 1]:
#         xlls = rng.random((self.nparticles,)) * 2 - 1
#         self.vlls = xlls * self.v
#         # In BEAMS3D, "MU" is defined as 0.5 * m * vperp^2 / B
#         self.mus = (
#             0.5
#             * self.m_kg
#             * (1 - xlls * xlls)
#             * self.v
#             * self.v
#             / modB_spl.ev(thetas, phis)
#         )

#     def create_outfile(self):
#         f = open(self.outfile, "w")
#         f.write("# nparticles\n")
#         f.write("{}\n".format(self.nparticles))
#         f.write("# theta, phi, v||, R, Z, mu, weight\n")
#         self.fmt_master = "{:24.15e}"
#         self.fmt_left = "{:<24.15e}"
#         fmt = 7 * self.fmt_master
#         fmt = fmt + "\n"
#         for j in range(self.nparticles):
#             f.write(
#                 fmt.format(
#                     self.thetas[j],
#                     self.phis[j],
#                     self.vlls[j],
#                     self.rs[j],
#                     self.zs[j],
#                     self.mus[j],
#                     1.0 / self.nparticles,
#                 )
#             )
#         f.close()

#     def create_beams3dfile(self):
#         # Now write the info needed by BEAMS3D:
#         nparticles = self.nparticles
#         fmt_master = self.fmt_master

#         f = open(self.beams3dfile, "w")

#         numstr = str(self.nparticles) + "*"

#         f.write("T_END_IN = " + numstr + "1.0d-5\n")
#         f.write("ZATOM_IN = " + numstr + "2.0d+0\n")
#         f.write("CHARGE_IN = " + numstr + self.fmt_left.format(2 * self.e_C) + "\n")
#         f.write("MASS_IN = " + numstr + self.fmt_left.format(self.m_kg) + "\n")
#         f.write("! Using volume Jacobian\n")
#         mystr = "R_START_IN = "
#         for j in range(nparticles):
#             mystr += fmt_master.format(self.rs[j])
#             if np.mod(j, 5) == 4:
#                 mystr += "\n"
#             f.write(mystr)
#             mystr = " "
#         f.write("\n")

#         mystr = "PHI_START_IN = "
#         for j in range(nparticles):
#             mystr += fmt_master.format(self.phis[j])
#             if np.mod(j, 5) == 4:
#                 mystr += "\n"
#             f.write(mystr)
#             mystr = " "
#         f.write("\n")

#         mystr = "Z_START_IN = "
#         for j in range(nparticles):
#             mystr += fmt_master.format(self.zs[j])
#             if np.mod(j, 5) == 4:
#                 mystr += "\n"
#             f.write(mystr)
#             mystr = " "
#         f.write("\n")

#         mystr = "VLL_START_IN = "
#         for j in range(nparticles):
#             mystr += fmt_master.format(self.vlls[j])
#             if np.mod(j, 5) == 4:
#                 mystr += "\n"
#             f.write(mystr)
#             mystr = " "
#         f.write("\n")

#         mystr = "MU_START_IN = "
#         for j in range(nparticles):
#             mystr += fmt_master.format(self.mus[j])
#             if np.mod(j, 5) == 4:
#                 mystr += "\n"
#             f.write(mystr)
#             mystr = " "
#         f.write("\n")

#         f.close()

#         print(
#             "min(sqrtg)={}, max(sqrtg)={}".format(
#                 np.min(self.sqrtg), np.max(self.sqrtg)
#             )
#         )

#     def plot_phitheta(self):
#         """
#         Make figure of the points
#         """
#         fig = plt.figure()
#         plt.contourf(self.phi, self.theta, self.sqrtg, 25)
#         plt.colorbar()
#         plt.xlabel("phi")
#         plt.ylabel("theta")
#         plt.plot(self.phis, self.thetas, ".k", ms=2)
#         plt.title("Color = sqrtg")

#         plt.show()


# if __name__ == "__main__":
#     print("Usage: ", __file__, " woutFile")
#     beams3D_from_vmec(filename=sys.argv[1])
