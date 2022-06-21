import neatf90
import numpy as np

z_vmec = np.array([0.5, 1.0, 1.0])

neatf90.init("wout.nc", 3, 3, 3, 1)
z_can = neatf90.init_orbit(z_vmec, 1.0)

print(z_can)

# Test: This should give zero up to numerical tolerance (about 1e-8 here)
theta_vmec, varphi_vmec = neatf90.can_to_vmec(z_can[0], z_can[1], z_can[2])
print(theta_vmec, varphi_vmec)