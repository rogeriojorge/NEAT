#!/usr/bin/env python3
from pysimple import new_vmec_stuff_mod as stuff  # isort:skip
from pysimple import simple, params, simple_main  # isort:skip

import matplotlib.pyplot as plt  # isort:skip
from scipy.io import netcdf  # isort:skip
import numpy as np  # isort:skip
import random  # isort:skip
import os  # isort:skip

## Input parameters
nparticles = 128
tfinal = 1e-3
s_initial = 0.5
B_scale = 2  # Scale the magnetic field
Aminor_scale = 2  # Scale the size of the plasma
wout_filename = "wout_W7X.nc"
wout_filename_prefix = f"{os.path.join(os.path.dirname(__file__))}/inputs/"
vparallel_over_v_min = -0.5
vparallel_over_v_max = 0.5

## Run SIMPLE
stuff.multharm = 3  # Fast but inaccurate splines
stuff.vmec_b_scale = B_scale
stuff.vmec_rz_scale = Aminor_scale
params.ntestpart = nparticles
params.trace_time = tfinal
params.contr_pp = -1e10  # Trace all passing passing
params.startmode = -1  # Manual start conditions

tracy = params.Tracer()
print(stuff.ns_s, stuff.ns_tp, stuff.multharm, params.integmode)
simple.init_field(
    tracy,
    wout_filename_prefix + wout_filename,
    stuff.ns_s,
    stuff.ns_tp,
    stuff.multharm,
    params.integmode,
)

params.params_init()
# s, th_vmec, ph_vmec, v/v0, v_par/v
net_file = netcdf.netcdf_file(wout_filename_prefix + wout_filename, "r", mmap=False)
nfp = net_file.variables["nfp"][()]
net_file.close()
params.zstart = (
    np.array(
        [
            [
                s_initial,
                random.uniform(0, 2 * np.pi),
                random.uniform(0, 2 * np.pi / nfp),
                1,
                random.uniform(vparallel_over_v_min, vparallel_over_v_max),
            ]
            for i in range(nparticles)
        ]
    )
    .reshape(nparticles, 5)
    .T
)

simple_main.run(tracy)

print(params.times_lost)

t = np.linspace(params.dtau / params.v0, params.trace_time, params.ntimstep)

plt.figure()
plt.semilogx(t, 1 - (params.confpart_pass + params.confpart_trap))
plt.xlim([1e-5, params.trace_time])
plt.xlabel("Time (s)")
plt.ylabel("Loss Fraction")

plt.figure()
condi = np.logical_and(params.times_lost > 0, params.times_lost < params.trace_time)
plt.semilogx(params.times_lost[condi], params.perp_inv[condi], "x")
plt.xlim([1e-5, params.trace_time])
plt.xlabel("Loss Time")
plt.ylabel("Perpendicular Invariant")
plt.show()
