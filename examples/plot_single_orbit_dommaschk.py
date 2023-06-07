#!/usr/bin/env python

import time

import numpy as np
import matplotlib.pyplot as plt
from simsopt.field import Dommaschk

from neat.fields import Dommaschk
from neat.tracing import ChargedParticle, ParticleOrbit

"""                                                                           
Trace the orbit of a single particle in a
quasisymmetric stellarator                 
"""

# Initialize an alpha particle at a radius = r_initial
r_initial = 1.03  # meters
theta_initial = np.pi/2  # initial poloidal angle
phi_initial = 0.03  # initial Z
B0 = 1  # Tesla, magnetic field on-axis
energy = 100000 # electron-volt
charge = 1  # times charge of proton
mass = 4  # times mass of proton
Lambda = 0.8  # = mu * B0 / energy
vpp_sign = -1  # initial sign of the parallel velocity, +1 or -1
nsamples = 10  # resolution in time
tfinal = 2e-4  # seconds

g_field = Dommaschk(m=[5], l=[2], coeff1=[1.4], coeff2=[1.4], B0=[B0])
#g_field = Dommaschk(m=[5,5], l=[2,4], coeff1=[1.4,19.25], coeff2=[1.4,0], B0=[B0,B0])
#g_field = Dommaschk(m=[5,5,5], l=[2,4,10], coeff1=[1.4,19.25,5.1e10], coeff2=[1.4,0,5.1e10], B0=[B0,B0,B0])
#g_field = Dommaschk(m=[5], l=[10], coeff1=[5.1e10], coeff2=[5.1e10], B0=[B0])
#g_field = Dommaschk(m=[5,5,5,5], l=[2,4,10,14], coeff1=[1.4,19.25,5.1e10,7e16], coeff2=[1.4,0,5.1e10,0], B0=[B0,B0,B0,B0])
#g_field = Dommaschk(m=[5], l=[12], coeff1=[5e18], coeff2=[5e18], B0=[B0])
#g_field = Dommaschk(m=[5], l=[14], coeff1=[5e22], coeff2=[5e22], B0=[B0])

""" Bfield = Dommaschk(mn=mn, coeffs=coeffs)
point = np.asarray([[0.9231, 0.8423, -0.1123]])
Bfield.set_points(point) """

# import matplotlib.pyplot as plt
# phi=np.linspace(0,10*np.pi,100)
# plt.plot(phi,g_field.B_mag(0.1, g_field.iota * phi, phi))
# plt.show()
# exit()

g_particle = ChargedParticle(
    r_initial=r_initial,
    theta_initial=theta_initial,
    phi_initial=phi_initial,
    energy=energy,
    Lambda=Lambda,
    charge=charge,
    mass=mass,
    vpp_sign=vpp_sign,
)

print("Starting particle tracer")
start_time = time.time()
g_orbit = ParticleOrbit(
    g_particle, g_field, nsamples=nsamples, tfinal=tfinal
)
total_time = time.time() - start_time
print(f"Finished in {total_time}s")

# g_orbit.magnetic_field_strength

# import matplotlib.pyplot as plt
# plt.plot(g_orbit.time, g_orbit.r_pos)
# plt.show()
# print(g_orbit.time , g_orbit.r_pos)

# print("Creating B contour plot")
# g_orbit.plot_orbit_contourB(show=False)

print("Creating parameter plot")
g_orbit.plot(show=True)

print("Creating 2D plot")
g_orbit.plot_orbit(show=True) 

# print("Creating 3D plot")
# g_orbit.plot_orbit_3d(show=True)

""" print("Creating animation plot")
g_orbit.plot_animation(show=True, save_movie=False) """

import numpy as np
import matplotlib.pyplot as plt

print("Creating 3D plot")
def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.. This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    Args:
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    
    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


rpos_cartesian = np.array(
    [
        g_orbit.rpos_cylindrical[0] * np.cos(g_orbit.rpos_cylindrical[1]),
        g_orbit.rpos_cylindrical[0] * np.sin(g_orbit.rpos_cylindrical[1]),
        g_orbit.rpos_cylindrical[2],
    ]
)

fig = plt.figure(figsize=(10, 3))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # Adjust the subplot layout
ax = fig.add_subplot(111, projection="3d")  # Use a single subplot
ax.plot3D(rpos_cartesian[0], rpos_cartesian[1], rpos_cartesian[2])
set_axes_equal(ax)

plt.tight_layout()  # Adjust spacing between subplots and labels
plt.show()





