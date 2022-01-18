#!/usr/bin/env python3
import time
import math
import sys
sys.path.append("..")
import inputs
from stell_repo import get_stel
import matplotlib.pyplot as plt
from functions import orbit
import numpy as np

## Stellarator to analyze
try:
    stel, name, r0, Lambda = get_stel(inputs.nphi,inputs.stel_id)
except Exception as e:
    try:
        stel = inputs.stel
        name = inputs.name
        r0   = inputs.r0
        Lambda = inputs.Lambda
    except Exception as e:
        sys.exit('Need to define in inputs.py either stel_id or stellarator parameters stel, name r0 and Lambda')

if __name__ == '__main__':
    print("---------------------------------")
    nN=stel.iota-stel.iotaN
    print("---------------------------------")
    print("Get particle orbit using gyronimo")
    result=[]
    start_time = time.time()
    for _phi0 in inputs.phi0:
        for _Lambda in Lambda:
            for _energy in inputs.energy:
                for _theta0 in inputs.theta0:
                    orbit_temp=orbit(stel,r0,_theta0,_phi0,inputs.charge,inputs.rhom,inputs.mass,_Lambda,_energy,inputs.nsamples,inputs.Tfinal,inputs.B20real)
                    if not math.isnan(orbit_temp[1][-1]):
                        result.append(orbit_temp)
    print("--- %s seconds ---" % (time.time() - start_time))

    # Check energy error for each orbit
    energy_error     = [np.log10(np.abs(res[4]-res[4][0]+1e-30)/res[4][0]) for res in result]
    max_energy_error = [max(eerror[3::]) for eerror in energy_error]
    print("Max Energy Error per orbit = ",max_energy_error)

    # Check canonical angular momentum error for each orbit
    p_phi_error     = [np.log10(np.abs(res[9]-res[9][0]+1e-30)/abs(res[9][0])) for res in result]
    max_p_phi_error = [max(perror[3::]) for perror in p_phi_error]
    print("Max Canonical Angular Momentum Error per orbit = ",max_p_phi_error)

    # Plot relevant quantities
    if inputs.showPlots==1:
        fig=plt.figure(figsize=(10,6))
        plt.subplot(3, 3, 1);[plt.plot(res[0],res[1])  for res in result];plt.xlabel('Time');plt.ylabel('r')
        plt.subplot(3, 3, 2);[plt.plot(res[0],res[4])  for res in result];plt.xlabel('Time');plt.ylabel('Energy')
        plt.subplot(3, 3, 3);[plt.plot(res[0],res[9])  for res in result];plt.xlabel('Time');plt.ylabel('p_phi')
        plt.subplot(3, 3, 4);[plt.plot(res[0],res[3])  for res in result];plt.xlabel('Time');plt.ylabel('Phi')
        plt.subplot(3, 3, 5);[plt.plot(res[0],res[2])  for res in result];plt.xlabel('Time');plt.ylabel('Theta')
        plt.subplot(3, 3, 6);[plt.plot(res[0],res[14]) for res in result];plt.xlabel('Time');plt.ylabel('V_parallel')

        plt.subplot(3, 3, 7)
        [plt.plot(res[1]*np.cos(res[2]),res[1]*np.sin(res[2])) for res in result]
        th=np.linspace(0,2*np.pi,100);plt.plot(r0*np.cos(th),r0*np.sin(th))
        plt.xlabel('r cos(theta)');plt.ylabel('r sin(theta)')

        points = np.array([[res[1],res[2],res[16]] for res in result]).transpose(0,2,1)
        rpos_cylindrical = np.array([stel.to_RZ(points[i]) for i in range(len(result))])
        rpos_cartesian = [[rpos_cylindrical[i][0]*np.cos(rpos_cylindrical[i][2]),rpos_cylindrical[i][0]*np.sin(rpos_cylindrical[i][2]),rpos_cylindrical[i][1]] for i in range(len(result))]
        boundary = np.array(stel.get_boundary(r=0.9*r0,nphi=90,ntheta=25,ntheta_fourier=16,mpol=8,ntor=15))

        plt.subplot(3, 3, 8)
        [plt.plot(rpos_cylindrical[i][0],rpos_cylindrical[i][1]) for i in range(len(result))]
        phi1dplot_RZ = np.linspace(0, 2 * np.pi / stel.nfp, 4, endpoint=False)
        [plt.plot(boundary[3,:,int(phi / (2 * np.pi) * 90)],boundary[2,:,int(phi / (2 * np.pi) * 90)]) for phi in phi1dplot_RZ]
        plt.xlabel('R');plt.ylabel('Z')

        plt.subplot(3, 3, 9)
        [plt.plot(res[0],res[15],label='gyronimo') for res in result]
        [plt.plot(res[0],stel.B_mag(res[1], res[2], res[3]),label='theory') for res in result]
        plt.xlabel('Time');plt.ylabel('Bfield');plt.legend()

        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # [ax.plot3D(rpos_cartesian[i][0],rpos_cartesian[i][1],rpos_cartesian[i][2]) for i in range(len(result))]
        # ax.plot_surface(boundary[0],boundary[1],boundary[2],alpha=0.5);set_axes_equal(ax);ax.set_axis_off()
        # ax.dist = 6.0
        # import mayavi.mlab as mlab
        # fig = mlab.figure(bgcolor=(1,1,1), size=(430,720))
        # [mlab.plot3d(rpos_cartesian[i][0],rpos_cartesian[i][1],rpos_cartesian[i][2], color=(0.5,0.5,0.5)) for i in range(len(result))]

        # ntheta=80
        # nphi=220
        # X_qsc, Y_qsc, Z_qsc, R_qsc = stel.get_boundary(r=0.95*r0, ntheta=ntheta, nphi=nphi)

        # theta1D = np.linspace(0, 2 * np.pi, ntheta)
        # phi1D = np.linspace(0, 2 * np.pi, nphi)
        # phi2D, theta2D = np.meshgrid(phi1D, theta1D)
        # Bmag = stel.B_mag(r0, theta2D, phi2D)

        # mlab.mesh(X_qsc, Y_qsc, Z_qsc, scalars=Bmag, colormap='viridis')
        # mlab.view(azimuth=0, elevation=0, distance=8.5, focalpoint=(-0.15,0,0), figure=fig)

        # mlab.show()
        plt.show()