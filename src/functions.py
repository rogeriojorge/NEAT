import NEAT

import numpy as np
import time
from scipy.interpolate import UnivariateSpline as spline
import matplotlib.pyplot as plt
from matplotlib import cm
from joblib import Parallel, delayed
from functools import partial

def check_energy_error(result):
    # Check energy error for each orbit
    energy_error     = [np.log10(np.abs(res[4]-res[4][0]+1e-30)/res[4][0]) for res in result]
    max_energy_error = [max(eerror[3::]) for eerror in energy_error]
    print("Max Energy Error per orbit = ",max_energy_error)

def check_canonical_angular_momentum_error(result):
    # Check canonical angular momentum error for each orbit
    p_phi_error     = [np.log10(np.abs(res[9]-res[9][0]+1e-30)/abs(res[9][0])) for res in result]
    max_p_phi_error = [max(perror[3::]) for perror in p_phi_error]
    print("Max Canonical Angular Momentum Error per orbit = ",max_p_phi_error)

def create_BEAMS3D_input(name,stel,rhom,mass,
            rSurf,zSurf,r0,theta0,phiCylindrical,
            phi0,charge,Tfinal,nsamples,rParticleVec,
            zParticleVec,phiParticleVec,result):
    print('Input for BEAMS3D')
    m_proton = 1.67262192369e-27
    e = 1.602176634e-19
    mu0 = 1.25663706212e-6
    Valfven = stel.B0/np.sqrt(mu0*rhom*m_proton*1.e+19)
    Ualfven = 0.5*m_proton*mass*Valfven*Valfven
    energySI = result[0][4][0]*Ualfven
    Bfield0  = result[0][15][0]
    print('  R_START_IN =',rSurf(r0,phi0[0],theta0[0]))
    print('  Z_START_IN =',zSurf(r0,phi0[0],theta0[0]))
    print('  PHI_START_IN =',phiCylindrical(phi0[0]))
    print('  CHARGE_IN =',e)
    print('  MASS_IN =',mass*m_proton)
    print('  ZATOM_IN =',charge)
    print('  T_END_IN =',Tfinal*stel.rc[0]/Valfven)
    print('  NPOINC =',nsamples)
    print('  VLL_START_IN =',Valfven*result[0][14][0])
    print('  MU_START_IN =',energySI/Bfield0)
    print('')
    import os
    os.chdir('results/'+name)
    np.savetxt('rParticleVec.txt',rParticleVec)
    np.savetxt('zParticleVec.txt',zParticleVec)
    np.savetxt('phiParticleVec.txt',phiParticleVec)
    np.savetxt('vparallel.txt',np.array(result[0][14]))
    os.chdir("..")
    os.chdir("..")

## Obtain particle orbit
def orbit(stel,r0,theta0,phi0,charge,rhom,mass,Lambda,energy,nsamples,Tfinal,B20real):
    if stel.order=='r1':
        stel.B20      = [0]*len(stel.varphi)
        stel.B20_mean = 0
        stel.G2       = 0
        stel.beta_1s  = 0
        stel.X20      = [0]*len(stel.varphi)
        stel.X2c      = [0]*len(stel.varphi)
        stel.X2s      = [0]*len(stel.varphi)
        stel.Y20      = [0]*len(stel.varphi)
        stel.Y2c      = [0]*len(stel.varphi)
        stel.Y2s      = [0]*len(stel.varphi)
        stel.Z20      = [0]*len(stel.varphi)
        stel.Z2c      = [0]*len(stel.varphi)
        stel.Z2s      = [0]*len(stel.varphi)

    if B20real==1:
        B20=np.append(stel.B20,stel.B20[0])
    else:
        B20=[stel.B20_mean]*(len(stel.varphi)+1)
    # Call Gyronimo
    sol = np.array(NEAT.gc_solver(int(stel.nfp), stel.d_l_d_varphi,
    stel.B0, stel.etabar, B20, stel.B2c, stel.B2s,
    stel.G0, stel.G2, stel.I2, stel.iota, stel.iotaN, stel.beta_1s,
    np.append(stel.varphi,2*np.pi/stel.nfp),
    np.append(stel.torsion,stel.torsion[0]),
    np.append(stel.curvature,stel.curvature[0]),
    np.append(stel.X1c,stel.X1c[0]),
    np.append(stel.X1s,stel.X1s[0]),
    np.append(stel.Y1c,stel.Y1c[0]),
    np.append(stel.Y1s,stel.Y1s[0]),
    np.append(stel.X20,stel.X20[0]),
    np.append(stel.X2c,stel.X2c[0]),
    np.append(stel.X2s,stel.X2s[0]),
    np.append(stel.Y20,stel.Y20[0]),
    np.append(stel.Y2c,stel.Y2c[0]),
    np.append(stel.Y2s,stel.Y2s[0]),
    np.append(stel.Z20,stel.Z20[0]),
    np.append(stel.Z2c,stel.Z2c[0]),
    np.append(stel.Z2s,stel.Z2s[0]),
    charge, rhom, mass, Lambda,
    energy, r0, theta0, phi0, nsamples, stel.rc[0], Tfinal))

    time      = sol[:,0]
    r_pos     = sol[:,1]
    theta_pos = sol[:,2]
    phi_pos   = sol[:,3]
    energy_parallel = sol[:,4]
    energy_perpendicular = sol[:,5]
    Bfield = sol[:,6]
    v_parallel = sol[:,7]
    rdot = sol[:,8]
    thetadot = sol[:,9]
    phidot = sol[:,10]
    
    total_energy = energy_parallel+energy_perpendicular
    vppdot = sol[:,11]
    m_proton = 1.67262192369e-27
    e = 1.602176634e-19
    mu0 = 1.25663706212e-6
    Valfven = 1/np.sqrt(mu0*rhom*m_proton*1.e+19)

    # p_phi = mass*m_proton*v_parallel*Valfven*stel.B0*(stel.G0+r_pos**2*(stel.G2+(stel.iota-stel.iotaN)*stel.I2))/Bfield-charge*e*r_pos**2*stel.B0/2*stel.iotaN
    p_phi1=mass*m_proton*v_parallel*Valfven*(stel.G0+r_pos**2*(stel.G2+(stel.iota-stel.iotaN)*stel.I2))/Bfield
    p_phi2=charge*e*r_pos**2*stel.B0/2*stel.iotaN
    p_phi=p_phi1-p_phi2
    # ratio=(p_phi1-np.mean(p_phi1))/(p_phi2-np.mean(p_phi2))
    # print(ratio[10])
    # print(" ")
    # plt.figure()
    # plt.plot(p_phi1-np.mean(p_phi1))
    # plt.plot(p_phi2-np.mean(p_phi2))
    # plt.plot(p_phi)
    # plt.show()
    # exit()
    return [time,r_pos,theta_pos,phi_pos,total_energy,theta0,phi0,Lambda,energy,p_phi,rdot,thetadot,phidot,vppdot,v_parallel,Bfield]

def rxyzOrbit(rSurf,zSurf,phiCylindrical,nfp,r_pos,theta_pos,phi_pos,i):
    rPar=rSurf(r_pos[i],phi_pos[i],theta_pos[i])
    zPar=zSurf(r_pos[i],phi_pos[i],theta_pos[i])
    phiPar=phiCylindrical(phi_pos[i])+phi_pos[i]-np.mod(phi_pos[i],2*np.pi/nfp)
    xPar=rPar*np.cos(phi_pos[i])
    yPar=rPar*np.sin(phi_pos[i])
    return rPar,zPar,xPar,yPar,phiPar

def orbit_RZ(stel,result,r0,step,ncores):
    print("---------------------------------")
    print("Get particle orbit in the (R,Z) plane")
    start_time = time.time()
    def Raxisf(phi): return sum([stel.rc[i]*np.cos(i*stel.nfp*phi) for i in range(len(stel.rc))])
    def Zaxisf(phi): return sum([stel.zs[i]*np.sin(i*stel.nfp*phi) for i in range(len(stel.zs))])
    def tangentR(phi):
        sp=spline(stel.varphi, stel.tangent_cylindrical[:,0], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/stel.nfp))
    def tangentZ(phi):
        sp=spline(stel.varphi, stel.tangent_cylindrical[:,2], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/stel.nfp))
    def normalR(phi):
        sp=spline(stel.varphi, stel.normal_cylindrical[:,0], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/stel.nfp))
    def normalZ(phi):
        sp=spline(stel.varphi, stel.normal_cylindrical[:,2], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/stel.nfp))
    def binormalR(phi):
        sp=spline(stel.varphi, stel.binormal_cylindrical[:,0], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/stel.nfp))
    def binormalZ(phi):
        sp=spline(stel.varphi, stel.binormal_cylindrical[:,2], k=3, s=0)
        return sp(np.mod(phi,2*np.pi/stel.nfp))
    def X1cF(phi):
        sp=spline(stel.varphi, stel.X1c, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/stel.nfp))
    def X1sF(phi):
        sp=spline(stel.varphi, stel.X1s, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/stel.nfp))
    def Y1cF(phi):
        sp=spline(stel.varphi, stel.Y1c, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/stel.nfp))
    def Y1sF(phi):
        sp=spline(stel.varphi, stel.Y1s, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/stel.nfp))
    def X20F(phi):
        sp=spline(stel.varphi, stel.X20, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/stel.nfp))
    def X2cF(phi):
        sp=spline(stel.varphi, stel.X2c, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/stel.nfp))
    def X2sF(phi):
        sp=spline(stel.varphi, stel.X2s, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/stel.nfp))
    def Y20F(phi):
        sp=spline(stel.varphi, stel.Y20, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/stel.nfp))
    def Y2cF(phi):
        sp=spline(stel.varphi, stel.Y2c, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/stel.nfp))
    def Y2sF(phi):
        sp=spline(stel.varphi, stel.Y2s, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/stel.nfp))
    def Z20F(phi):
        sp=spline(stel.varphi, stel.Z20, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/stel.nfp))
    def Z2cF(phi):
        sp=spline(stel.varphi, stel.Z2c, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/stel.nfp))
    def Z2sF(phi):
        sp=spline(stel.varphi, stel.Z2s, k=3, s=0)
        return sp(np.mod(phi,2*np.pi/stel.nfp))
    def rSurf(r,phi,theta):
        thetaN = theta-0*(stel.iota-stel.iotaN)*phi
        r00= Raxisf(phiCylindrical(phi))
        r1 = (          X1cF(phi)*np.cos(thetaN  )+X1sF(phi)*np.sin(thetaN)  )*normalR(phi)+(          Y1cF(phi)*np.cos(thetaN  )+Y1sF(phi)*np.sin(thetaN)  )*binormalR(phi)
        r2 = (X20F(phi)+X2cF(phi)*np.cos(2*thetaN)+X2sF(phi)*np.sin(2*thetaN))*normalR(phi)+(Y20F(phi)+Y2cF(phi)*np.cos(2*thetaN)+Y2sF(phi)*np.sin(2*thetaN))*binormalR(phi)+(Z20F(phi)+Z2cF(phi)*np.cos(2*thetaN)+Z2sF(phi)*np.sin(2*thetaN))*tangentR(phi)
        return r00+r*r1+r**2*r2
    def zSurf(r,phi,theta):
        thetaN = theta-0*(stel.iota-stel.iotaN)*phi
        z0 = Zaxisf(phiCylindrical(phi))
        z1 = (          X1cF(phi)*np.cos(thetaN)  +X1sF(phi)*np.sin(thetaN)  )*normalZ(phi)+(          Y1cF(phi)*np.cos(thetaN)  +Y1sF(phi)*np.sin(thetaN)  )*binormalZ(phi)
        z2 = (X20F(phi)+X2cF(phi)*np.cos(2*thetaN)+X2sF(phi)*np.sin(2*thetaN))*normalZ(phi)+(Y20F(phi)+Y2cF(phi)*np.cos(2*thetaN)+Y2sF(phi)*np.sin(2*thetaN))*binormalZ(phi)+(Z20F(phi)+Z2cF(phi)*np.cos(2*thetaN)+Z2sF(phi)*np.sin(2*thetaN))*tangentZ(phi)
        return z0+r*z1+r**2*z2
    def phiCylindrical(varphi):
        sp=spline(stel.varphi, stel.phi, k=3, s=0)
        return sp(np.mod(varphi,2*np.pi/stel.nfp))
    R_pos     = [res[1] for res in result]
    Theta_pos = [res[2] for res in result]
    Phi_pos   = [res[3] for res in result]
    xParticleVec=[]
    yParticleVec=[]
    zParticleVec=[]
    rParticleVec=[]
    phiParticleVec=[]
    for j in range(len(R_pos)):
        r_pos     = R_pos[j][0:len(R_pos[j]):step]
        theta_pos = Theta_pos[j][0:len(R_pos[j]):step]
        phi_pos   = Phi_pos[j][0:len(R_pos[j]):step]
        phimod=np.mod(phi_pos,2*np.pi)
        iPhiPass0=np.array([])
        for i in range(len(phi_pos)-1):
            i=i+1
            if phimod[i]<phimod[i-1]-np.pi:
                iPhiPass0=np.append(iPhiPass0,i)
        iPhiPass0=iPhiPass0.astype(int)
        rxyzOrbit_temp = partial(rxyzOrbit,rSurf,zSurf,phiCylindrical,stel.nfp,r_pos,theta_pos,phi_pos)
        rParticle, zParticle, xParticle, yParticle, phiParticle = np.array(Parallel(n_jobs=ncores)(delayed(rxyzOrbit_temp)(i) for i in range(len(r_pos)))).T
        plt.plot(rParticle,zParticle,linewidth=0.5,zorder=-1)
        plt.scatter(rParticle[iPhiPass0],zParticle[iPhiPass0],c='r',s=8,zorder=1) # plot when particle it passes at phi=0
        xParticleVec.append(xParticle)
        yParticleVec.append(yParticle)
        zParticleVec.append(zParticle)
        rParticleVec.append(rParticle)
        phiParticleVec.append(phiParticle)
    print("--- %s seconds ---" % (time.time() - start_time))

    print("---------------------------------")
    print("Get stellarator boundary in the (R,Z) plane")
    start_time = time.time()
    ntheta=40; nphi=9
    theta = np.linspace(0,2*np.pi,ntheta)
    phi1D = np.linspace(0,2*np.pi/stel.nfp,nphi)
    for phi in phi1D:
        rSurfi=rSurf(r0,phi,theta)
        zSurfi=zSurf(r0,phi,theta)
        plt.plot(rSurfi,zSurfi,linewidth=2,color='black')
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.gca().set_aspect('equal')
    plt.xlabel('R (meters)')
    plt.ylabel('Z')
    return xParticleVec,yParticleVec,rParticleVec,zParticleVec,phiParticleVec,rSurf,zSurf,phiCylindrical

def orbit_3D(stel,r0,xParticleVec,yParticleVec,zParticleVec,rSurf,zSurf,ntheta,nphi,ax):
    print("---------------------------------")
    print("Get particle orbit and stellarator boundary in 3D")
    start_time = time.time()
    for i in range(len(xParticleVec)):
        ax.plot(xParticleVec[i], yParticleVec[i], zParticleVec[i],'-o', markersize=0.7, linewidth=0.7)

    def Bf(r,phi,theta):
        thetaN = theta-(stel.iota-stel.iotaN)*phi
        return stel.B0*(1+r*stel.etabar*np.cos(thetaN))+r*r*(stel.B20_mean+stel.B2c*np.cos(2*thetaN)+stel.B2s*np.sin(2*thetaN))
    theta1D = np.linspace(0,2*np.pi,ntheta)
    phi1D   = np.linspace(0,2*np.pi,nphi)
    Xsurf = np.zeros((ntheta,nphi))
    Ysurf = np.zeros((ntheta,nphi))
    Zsurf = np.zeros((ntheta,nphi))
    Bmag  = np.zeros((ntheta,nphi))
    for countT,th in enumerate(theta1D):
        for countP,ph in enumerate(phi1D):
            rs=rSurf(r0,ph,th)
            zs=zSurf(r0,ph,th)
            Xsurf[countT,countP]=rs*np.cos(ph)
            Ysurf[countT,countP]=rs*np.sin(ph)
            Zsurf[countT,countP]=zs
            Bmag[countT,countP] =Bf(r0,ph,th)
    B_rescaled = (Bmag - Bmag.min()) / (Bmag.max() - Bmag.min())
    ax.plot_surface(Xsurf, Ysurf, Zsurf, facecolors = cm.jet(B_rescaled), rstride=1, cstride=1, antialiased=False, linewidth=0, alpha=0.25)
    ax.auto_scale_xyz([Xsurf.min(), Xsurf.max()], [Xsurf.min(), Xsurf.max()], [Xsurf.min(), Xsurf.max()])   
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    plt.tight_layout()
    print("--- %s seconds ---" % (time.time() - start_time))
    return Xsurf, Ysurf, Zsurf, B_rescaled
