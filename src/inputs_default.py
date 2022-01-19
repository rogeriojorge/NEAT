from qsc import Qsc

nphi = 251 # resolution of the magnetic axis
theta0   = [3.14]
phi0     = [0.01]
charge   = 1
rhom     = 1
mass     = 1
energy   = [4e4]
nsamples = 1000
Tfinal   = 1000

B20real   = True
makePlots = 1
showPlots = 0
Animation = 0
SaveMovie = 0

nstep  = 1 # step size in the time series for the particle orbit (step=1 show all time steps)
ncores = 6 # number of cores used in the parallelization to unpack orbit position(t)

ntheta = 30 # poloidal resolution for the 3D plot
# nphi   = 60 # ##CLASH WITH PREVOUS nph #toroidal resolution for the 3D plot

boundaryR0 = 0.09

# stel_id = 0
paper_case = 2
stel = Qsc.from_paper(paper_case,nphi=nphi,B0=3)
name = 'paper_r2_5.'+str(paper_case) # name of results folder and files
r0       = 0.005
Lambda   = [0.2]

makePlots = 1
savePlots = 1