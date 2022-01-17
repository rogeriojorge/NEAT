import time
import math
import inputs
from stell_repo import get_stel
import matplotlib.pyplot as plt

## Stellarator to analyze
stel, name, r0, Lambda = get_stel(inputs.nphi,inputs.stel_id)

if __name__ == '__main__':
    print("---------------------------------")
    if stel.iotaN-stel.iota == 0:
        print('Quasi-axisymmetric stellarator with iota =',stel.iota)
    else:
        print('Quasi-helically symmetric stellarator with iota =',stel.iota)
    print("---------------------------------")
    print("Get particle orbit using gyronimo")
    result=[]
    # pool = multiprocessing.Pool(4)
    start_time = time.time()
    for _phi0 in inputs.phi0:
        for _Lambda in Lambda:
            for _energy in inputs.energy:
                for _theta0 in inputs.theta0:
                    orbit_temp=orbit(stel,r0,_theta0,_phi0,inputs.charge,inputs.rhom,inputs.mass,_Lambda,_energy,inputs.nsamples,inputs.Tfinal,inputs.B20real)
                    if not math.isnan(orbit_temp[1][-1]):
                        result.append(orbit_temp)
    print("--- %s seconds ---" % (time.time() - start_time))

    # Show plot
    if input.showPlots==1:
        plt.show()