#!/usr/bin/env python3

#%%
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from pysimple import new_vmec_stuff_mod as stuff
from pysimple import params as p
from pysimple import simple, simple_main
from scipy.io import netcdf

## Input parameters
nparticles = 128
tfinal = 1e-3
s_initial = 0.6
B_scale = 1  # Scale the magnetic field
Aminor_scale = 1  # Scale the size of the plasma
wout_filename = "wout_ESTELL.nc"
wout_filename_prefix = f"{os.path.join(os.path.dirname(__file__))}/inputs/"
vparallel_over_v_min = -0.5
vparallel_over_v_max = 0.5


## Run the SIMPLE code
for item in dir(p):
    if item.startswith("__"):
        continue
    try:
        print(f"{item}: {getattr(p, item)}")
    except:
        print(f"{item}: NULL")

stuff.multharm = 3  # Fast but inaccurate splines
stuff.vmec_b_scale = B_scale
stuff.vmec_rz_scale = Aminor_scale
p.ntestpart = nparticles
p.trace_time = tfinal
p.contr_pp = -1e10  # Trace all passing passing
p.startmode = -1  # Manual start conditions

tracy = p.Tracer()
print(stuff.ns_s, stuff.ns_tp, stuff.multharm, p.integmode)
simple.init_field(
    tracy,
    wout_filename_prefix + wout_filename,
    stuff.ns_s,
    stuff.ns_tp,
    stuff.multharm,
    p.integmode,
)


p.params_init()
#%%
# s, th_vmec, ph_vmec, v/v0, v_par/v
net_file = netcdf.netcdf_file(wout_filename_prefix + wout_filename, "r", mmap=False)
nfp = net_file.variables["nfp"][()]
net_file.close()
p.zstart = (
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
#%%
simple_main.run(tracy)

print(p.times_lost)

t = np.linspace(p.dtau / p.v0, p.trace_time, p.ntimstep)

plt.figure()
plt.semilogx(t, 1 - (p.confpart_pass + p.confpart_trap))
plt.xlim([1e-5, p.trace_time])
plt.xlabel("Time (s)")
plt.ylabel("Loss Fraction")

plt.figure()
condi = np.logical_and(p.times_lost > 0, p.times_lost < p.trace_time)
plt.semilogx(p.times_lost[condi], p.perp_inv[condi], "x")
plt.xlim([1e-5, p.trace_time])
plt.xlabel("Loss Time")
plt.ylabel("Perpendicular Invariant")
plt.show()
