#!/usr/bin/env python3
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from neat.fields import StellnaQS
from neat.tracing import ChargedParticle, ParticleOrbit
from neat.plotting import butter_lowpass_filter, butter_lowpass_filter2

step_i = 0.25
s_in = 0.5
nfp = 4
s_initials = np.round(np.arange(s_in, 0.96, step_i), 2)  # psi/psi_a for vmec
lambdas = np.round(np.arange(0.96, 0.999, 0.01), 2)  # = mu * B0 / energy
theta_initials = np.round(np.arange(0, 2 * np.pi, 2 * np.pi / 2), 2)  # initial poloidal angle (vartheta=theta_Boozer - N phi_Boozer)
phi_initials = np.round(np.arange(0, 2 * np.pi / nfp, 2 * np.pi / nfp / 8), 2)  # initial toroidal angle (cylindrical on axis)
B0 = 5.3267  # Tesla, magnetic field on-axis (ARIES-CS)
energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
nsamples = 80000  # resolution in time
tfinal = 1e-2  # seconds
constant_b20 = False  # use a constant B20 (mean value) or the real function
Rmajor_ARIES = 7.7495 * 2
Rminor_ARIES = 1.7044
r_avg = Rminor_ARIES
stellarator = "2022 QH nfp4 well"
g_field_basis = StellnaQS.from_paper(stellarator, B0=B0, nphi=101)
g_field = StellnaQS(
    rc=g_field_basis.rc * Rmajor_ARIES,
    zs=g_field_basis.zs * Rmajor_ARIES,
    etabar=g_field_basis.etabar / Rmajor_ARIES,
    B2c=g_field_basis.B2c * (B0 / Rmajor_ARIES / Rmajor_ARIES),
    B0=B0,
    nfp=g_field_basis.nfp,
    order='r3',
    nphi=101
)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    print("Starting orbit scan")
start_time = time.time()

def compute_orbit(params):
    lambdas_in, s_initial, theta_initial, phi_initial, vpp_sign = params
    r_initial=r_avg*np.sqrt(s_initial)
    theta_initial-=(g_field.iota-g_field.iotaN)*phi_initial
    particle = ChargedParticle(
        r_initial=r_initial,
        theta_initial=theta_initial,
        phi_initial=phi_initial,
        energy=energy,
        Lambda=lambdas_in,
        charge=charge,
        mass=mass,
        vpp_sign=vpp_sign
    )
    orbit = ParticleOrbit(particle, g_field, nsamples=nsamples, tfinal=tfinal, constant_b20=constant_b20)
    # orbit.plot_orbit_contourB(savefig=os.path.join(folder, features_str + 'B_neat.png'), show=False)
    orbit.plot(r_minor=r_avg, savefig=os.path.join(folder, features_str + 'param_neat.png'), show=False)
    # orbit.plot_orbit_3d(r_surface=r_avg, savefig=os.path.join(folder, features_str + 'orbit3d.png'), show=False)
    plt.close()

params = [
    (lambdas_in, s_initial, theta_initial, phi_initial, vpp_sign)
    for lambdas_in in lambdas
    for s_initial in s_initials
    for theta_initial in theta_initials
    for phi_initial in phi_initials
    for vpp_sign in [+1.0, -1.0]
]

local_params =  [tuple(arr) for arr in np.array_split(params, size)[rank]]

folder = stellarator + '_QSC2_A=' + str(Rmajor_ARIES / r_avg)
os.makedirs(folder, exist_ok=True)

for index, param in enumerate(local_params):
    lambda_in, s_initial, theta_initial, phi_initial, vpp_sign = param
    lambda_str = str(lambda_in)
    s_str = str(s_initial)
    theta_str = str(theta_initial)
    varphi_str = str(phi_initial)
    sign_str = '1' if vpp_sign == +1 else '-1'
    features_str = f'{lambda_str}_{s_str}_{theta_str}_{varphi_str}_{sign_str}_'
    compute_orbit(param)

total_time = comm.reduce(time.time() - start_time, op=MPI.MAX, root=0)
if rank == 0:
    print(f"Finished in {total_time}s")
    print(f'Plots saved in {folder} after {total_time}s')
