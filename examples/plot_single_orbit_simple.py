#!/usr/bin/env python3

from optparse import AmbiguousOptionError
from pysimple import params, simple, orbit_symplectic, vmec_to_can, can_to_vmec, new_vmec_stuff_mod as vmec_stuff
import numpy as np
import matplotlib.pyplot as plt
import os

## Input parameters
nsamples = 30000
s_initial = 0.4
theta_initial = np.pi/2
phi_initial = 0
vparallel_over_v_initial = -0.05
B_scale = 5
Aminor_scale = 10
wout_filename = "wout_ESTELL.nc"
# wout_filename = "wout_ITER.nc"
# wout_filename = "wout_W7X.nc"
wout_filename_prefix = f"{os.path.join(os.path.dirname(__file__))}/inputs/"

tracy = params.Tracer()
vmec_stuff.vmec_b_scale = B_scale
vmec_stuff.vmec_rz_scale = Aminor_scale

simple.init_field(tracy, wout_filename_prefix+wout_filename, 3, 3, 3, 1)
simple.init_params(tracy, 2, 4, 3.5e6, 256, 1, 1e-13)

# Initial conditions
# z0_vmec = np.array([0.5, 0.3, 0.2, 1.0, 0.1])   # s, th, ph, v/v_th, v_par/v
z0_vmec = np.array([s_initial, theta_initial, phi_initial, 1.0, vparallel_over_v_initial])    # s, th_c, ph_c, v/v_th (should be 1 as we're non relativstic), v_par/v
z0_can = z0_vmec.copy()  # s, th_c, ph_c, v/v_th, v_par/v

z0_can[1:3] = vmec_to_can(z0_vmec[0], z0_vmec[1], z0_vmec[2])

simple.init_integrator(tracy, z0_can)

print(f'B = {tracy.f.bmod}')

nt = nsamples
z_integ = np.zeros([nt, 4])  # s, th_c, ph_c, p_phi
z_vmec = np.zeros([nt, 5])  # s, th, ph, v/v_th, v_par/v
z_integ[0,:] = tracy.si.z
z_vmec[0,:] = z0_vmec

for kt in range(nt-1):
    orbit_symplectic.orbit_timestep_sympl(tracy.si, tracy.f)
    z_integ[kt+1, :] = tracy.si.z
    z_vmec[kt+1, 0] = z_integ[kt+1, 0]
    z_vmec[kt+1, 1:3] = can_to_vmec(
        z_integ[kt+1, 0], z_integ[kt+1, 1], z_integ[kt+1, 2])
    z_vmec[kt+1, 3] = np.sqrt(tracy.f.mu*tracy.f.bmod+0.5*tracy.f.vpar**2)
    z_vmec[kt+1, 4] = tracy.f.vpar/(z_vmec[kt+1, 3]*np.sqrt(2))

fig = plt.figure()
ax = fig.add_subplot(111,aspect='equal')
plt.plot(z_integ[:, 0]*np.cos(z_integ[:, 1]), z_vmec[:, 0]*np.sin(z_integ[:, 1]))
plt.xlabel(r'$s \cdot cos(\theta)$')
plt.ylabel(r'$s \cdot sin(\theta)$')
plt.title('Poloidal orbit topology')

plt.figure()
plt.plot(z_vmec[:, 3])
plt.plot(z_vmec[:, 4])
plt.xlabel('Timestep')
plt.ylabel('Normalized velocity')
plt.legend([r'$v/v_0$', r'$v_{parallel}/v$'])
plt.title('Velocities over time')

plt.show()
