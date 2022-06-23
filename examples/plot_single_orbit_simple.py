#!/usr/bin/env python3

from pysimple import simple, orbit_symplectic
import numpy as np
import matplotlib.pyplot as plt
import os

wout_filename = "wout_ESTELL.nc"
# wout_filename = "wout_ITER.nc"
# wout_filename = "wout_W7X.nc"
wout_filename_prefix = f"{os.path.join(os.path.dirname(__file__))}/inputs/"

tracy = simple.Tracer()

simple.init_field(tracy, wout_filename_prefix+wout_filename, 5, 5, 7, 1)
simple.init_params(tracy, 2, 4, 3.5e6, 256, 1, 1e-13)

z0 = np.array([0.5, 0.0, 0.0, 1.0, 0.1])    # s, th_c, ph_c, v/v_th, v_par/v
simple.init_integrator(tracy, z0)

print(f'B = {tracy.f.bmod}')

nt = 10000
z = np.empty([nt, 4])  # s, th_c, ph_c, p_phi
z[0,:] = tracy.si.z

for kt in range(nt-1):
    orbit_symplectic.orbit_timestep_sympl(tracy.si, tracy.f)
    z[kt+1, :] = tracy.si.z

plt.plot(z[:, 0]*np.cos(z[:, 1]), z[:, 0]*np.sin(z[:, 1]))
plt.show()
