#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from mpl_toolkits import mplot3d

from neatpp import vmectrace

# Plotting orbit tracing
x = np.array(
    vmectrace(
        mass=1.0,
        charge=1.0,
        energy=1.0,
        s0=0.1,
        theta0=0.5,
        phi0=0.01,
        Lambda=0.8,
        Tfinal=1.0,
        nsamples=1000,
        vmec_file="wout.nc",
    )
)
time = x[:, 0]
R = x[:, 1]
phi = x[:, 2]
Z = x[:, 3]
Bfield = x[:, 6]
fig = plt.subplots()
ax = plt.axes(projection="3d")
ax.plot3D(R * np.cos(phi), R * np.sin(phi), Z, label="Particle 1")
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.xlabel("X")
plt.ylabel("Y")
# plt.zlabel('Z')
plt.title("First particle tracing")
plt.legend()
# plt.figure()
# plt.plot(time,Bfield)
# plt.xlabel('time')
# plt.ylabel('B')
plt.show()
