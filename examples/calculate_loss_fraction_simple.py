#!/usr/bin/env python3

from pysimple import simple, simple_main, params as p, new_vmec_stuff_mod as vmec_stuff
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
from scipy.io import netcdf
import os


## Input parameters
nparticles = 64
tfinal = 1e-2
s_initial = 0.5
B_scale = 2 # Scale the magnetic field 
Aminor_scale = 2 # Scale the size of the plasma
wout_filename = "wout_W7X.nc"
wout_filename_prefix = f"{os.path.join(os.path.dirname(__file__))}/inputs/"

## Run the SIMPLE code
for item in dir(p):
    if item.startswith("__"): continue
    try:
        print(f'{item}: {getattr(p, item)}')
    except:
        print(f'{item}: NULL')

vmec_stuff.multharm = 3     # Fast but inaccurate splines
vmec_stuff.vmec_b_scale = B_scale
vmec_stuff.vmec_rz_scale = Aminor_scale
p.ntestpart = nparticles
p.trace_time = tfinal
p.contr_pp = -1     # Trace all passing passing
p.startmode = -1       # Manual start conditions

tracy = p.Tracer()

simple.init_field(tracy, wout_filename_prefix+wout_filename,
    vmec_stuff.ns_s, vmec_stuff.ns_tp, vmec_stuff.multharm, p.integmode)

print(vmec_stuff.ns_s, vmec_stuff.ns_tp, vmec_stuff.multharm, p.integmode)

p.params_init()
#%%
# s, th_vmec, ph_vmec, v/v0, v_par/v
net_file = netcdf.netcdf_file(wout_filename_prefix+wout_filename, "r", mmap=False)
nfp = net_file.variables["nfp"][()]
net_file.close()
p.zstart = np.array([[s_initial, uniform(0,2*np.pi), uniform(0,2*np.pi/nfp), 1, uniform(-1,1)] for i in range(nparticles)]).reshape(nparticles, 5).T

#%%
simple_main.run(tracy)

print(p.times_lost)

t = np.linspace(p.dtau/p.v0, p.trace_time, p.ntimstep)

plt.figure()
plt.semilogx(t, 1 - (p.confpart_pass + p.confpart_trap))
plt.xlim([1e-6, p.trace_time])
plt.xlabel('Time (s)')
plt.ylabel('Loss Fraction')

plt.figure()
condi = np.logical_and(p.times_lost > 0, p.times_lost < p.trace_time)
plt.semilogx(p.times_lost[condi], p.perp_inv[condi], 'x')
plt.xlim([1e-6, p.trace_time])
plt.xlabel('loss time')
plt.ylabel('perpendicular invariant')
plt.show()

# %%
