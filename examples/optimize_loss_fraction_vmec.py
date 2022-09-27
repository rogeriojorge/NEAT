# from simsopt import make_optimizable
# from simsopt.mhd import Vmec
# from simsopt.util import MpiPartition
# from simsopt.solve import least_squares_mpi_solve
# from simsopt.objectives import LeastSquaresProblem
# from neat.fields import Simple
# from neat.tracing import ChargedParticleEnsemble, ParticleEnsembleOrbit_Simple
# from scipy.optimize import dual_annealing

# mpi = MpiPartition()

# MAXITER = 500
# max_modes = 1
# aspect_ratio_target = 7
# s_initial = 0.3  # initial normalized toroidal magnetic flux (radial VMEC coordinate)
# nparticles = 600  # number of particles
# tfinal = 6e-5  # total time of tracing in seconds
# nsamples = 1500 # number of time steps
# multharm = 3 # angular grid factor
# ns_s = 3 # spline order over s
# ns_tp = 3 # spline order over theta and phi
# nper = 400 # number of periods for initial field line
# npoiper = 150 # number of points per period on this field line
# npoiper2 = 120 # points per period for integrator step
# notrace_passing = 0 # if 1 skips tracing of passing particles, else traces them

# vmec = Vmec(filename, mpi=mpi, verbose=False)
# vmec.keep_all_files = True
# surf = vmec.boundary
# g_particle = ChargedParticleEnsemble(r_initial=s_initial)

# def EPcostFunction(v: Vmec):
#     v.run()
#     B_scale = 5.7/v.wout.b0  # Scale the magnetic field by a factor
#     Aminor_scale = 1.7/v.wout.Aminor_p  # Scale the machine size by a factor
#     g_field_temp = Simple(wout_filename=v.output_file, B_scale=B_scale, Aminor_scale=Aminor_scale, multharm=multharm,ns_s=ns_s,ns_tp=ns_tp)
#     for j in range(0,3): # Try three times the same orbits, if not able continue
#         while True:
#             try:
#                 g_orbits_temp = ParticleEnsembleOrbit_Simple(g_particle,g_field_temp,tfinal=tfinal,nparticles=nparticles,nsamples=nsamples,notrace_passing=notrace_passing,nper=nper,npoiper=npoiper,npoiper2=npoiper2)
#                 loss_fraction = g_orbits_temp.total_particles_lost
#             except ValueError as error_print:
#                 print(f'Try {j} of ParticleEnsembleOrbit_Simple gave error:',error_print)
#                 continue
#             break
#     return loss_fraction

# optEP = make_optimizable(EPcostFunction, vmec)
# opt_tuple = [(vmec.aspect, aspect_ratio_target, 1), (optEP.J, 0, 1)]

#     elif optimizer == 'least_squares_diff':
#         least_squares_mpi_solve(prob, mpi, grad=True, rel_step=diff_rel_step, abs_step=diff_abs_step, max_nfev=MAXITER)
#     elif optimizer == 'dual_annealing':
#         initial_temp = 1000
#         visit = 2.0
#         no_local_search = False
#         # bounds = [(np.max([-10*np.abs(dof),-0.21]),np.min([0.21,10*np.abs(dof)])) for dof in dofs]
#         bounds = [(-0.25,0.25) for _ in dofs]
#         res = dual_annealing(fun, bounds=bounds, maxiter=MAXITER, initial_temp=initial_temp,visit=visit, no_local_search=no_local_search, x0=dofs)