#!/usr/bin/env python3
from pysimple import new_vmec_stuff_mod as stuff  # isort:skip
from pysimple import simple, params  # isort:skip
from pysimple import can_to_vmec, orbit_symplectic, vmec_to_can, vmec_to_cyl

import matplotlib.pyplot as plt  # isort:skip
from mpl_toolkits import mplot3d  # isort:skip
import numpy as np  # isort:skip
import os  # isort:skip

## Input parameters
nsamples = 5000
s_initial = 0.5
theta_initial = np.pi / 2
phi_initial = 0.1
vparallel_over_v_initial = -0.2
B_scale = 2  # Scale the magnetic field
Aminor_scale = 2  # Scale the size of the plasma
wout_filename = "wout_W7X.nc"
wout_filename_prefix = f"{os.path.join(os.path.dirname(__file__))}/inputs/"

tracy = params.Tracer()
stuff.vmec_b_scale = B_scale
stuff.vmec_rz_scale = Aminor_scale
stuff.multharm = 3  # Fast but inaccurate splines
stuff.ns_s = 3
stuff.ns_tp = 3

simple.init_field(
    tracy,
    wout_filename_prefix + wout_filename,
    stuff.ns_s,
    stuff.ns_tp,
    stuff.multharm,
    params.integmode,
)
simple.init_params(tracy, 2, 4, 3.5e6, 256, 1, 1e-13)

# s, th, ph, v/v_th, v_par/v
z0_vmec = np.array(
    [s_initial, theta_initial, phi_initial, 1.0, vparallel_over_v_initial]
)
z0_can = z0_vmec.copy()

z0_can[1:3] = vmec_to_can(z0_vmec[0], z0_vmec[1], z0_vmec[2])

simple.init_integrator(tracy, z0_can)

print(f"B = {tracy.f.bmod}")

nt = nsamples
z_integ = np.zeros([nt, 4])  # s, th_c, ph_c, p_phi
z_vmec = np.zeros([nt, 5])  # s, th, ph, v/v_th, v_par/v
z_cyl = np.zeros([nt, 3])
z_integ[0, :] = tracy.si.z
z_vmec[0, :] = z0_vmec
z_cyl[0, :2] = vmec_to_cyl(z_vmec[0, 0], z_vmec[0, 1], z_vmec[0, 2])
z_cyl[0, 2] = z_vmec[0, 2]

for kt in range(nt - 1):
    orbit_symplectic.orbit_timestep_sympl(tracy.si, tracy.f)
    z_integ[kt + 1, :] = tracy.si.z
    z_vmec[kt + 1, 0] = z_integ[kt + 1, 0]
    z_vmec[kt + 1, 1:3] = can_to_vmec(
        z_integ[kt + 1, 0], z_integ[kt + 1, 1], z_integ[kt + 1, 2]
    )
    z_vmec[kt + 1, 3] = np.sqrt(tracy.f.mu * tracy.f.bmod + 0.5 * tracy.f.vpar**2)
    z_vmec[kt + 1, 4] = tracy.f.vpar / (z_vmec[kt + 1, 3] * np.sqrt(2))
    z_cyl[kt + 1, :2] = vmec_to_cyl(
        z_vmec[kt + 1, 0], z_vmec[kt + 1, 1], z_vmec[kt + 1, 2]
    )
    z_cyl[kt + 1, 2] = z_vmec[kt + 1, 2]

plt.figure()
plt.plot(z_vmec[:, 0] * np.cos(z_vmec[:, 1]), z_vmec[:, 0] * np.sin(z_vmec[:, 1]))
plt.xlabel(r"$s ~ \cos(\theta)$")
plt.ylabel(r"$s ~ \sin(\theta)$")
plt.title("Poloidal orbit topology")


plt.figure()
# plt.plot(z_vmec[:, 3])
plt.plot(z_vmec[:, 4])
plt.xlabel("Time (s)")
plt.ylabel(r"$v_\parallel/v$")
# plt.legend(['v/v_0', 'v_par/v'])
# plt.title('Velocities over time')

# Poloidal orbit in RZ
plt.figure()
plt.plot(z_cyl[:, 0], z_cyl[:, 1])
plt.xlabel("R")
plt.ylabel("Z")
plt.title("Poloidal orbit projection")

# 3D orbit in RZ
plt.figure()
ax = plt.axes(projection="3d")
ax.plot(
    z_cyl[:, 0] * np.cos(z_cyl[:, 2]), z_cyl[:, 0] * np.sin(z_cyl[:, 2]), z_cyl[:, 1]
)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Orbit in real space")
ax.set_xlim(-1400, 1400)
ax.set_ylim(-1400, 1400)
ax.set_zlim(-1400, 1400)

plt.show()
