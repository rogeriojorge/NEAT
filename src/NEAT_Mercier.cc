// // // NEAT: NEar-Axis sTellarator Particle Tracer
// This file is a python Wrapper
// It allows gyronimo to be called directly from Python
// Rogerio Jorge, July 2021, Greifswald
// // //
// Command to compile NEAT on a Macbook with gsl pre-installed with macports
// g++ -O2 -Wall -shared -std=c++20 -undefined dynamic_lookup $(python3 -m pybind11 --includes) -I/opt/local/include -L/opt/local/lib -lgsl -lblas -L../build -lgyronimo -I../include -isysroot`xcrun --show-sdk-path` NEAT_Mercier.cpp -o NEAT_Mercier.so

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gyronimo/interpolators/cubic_gsl.hh>
#include <gyronimo/metrics/metric_mercier.hh>
#include <gyronimo/core/dblock.hh>
#include <vector>
#include <gyronimo/fields/equilibrium_mercier.hh>
#include <gyronimo/dynamics/guiding_centre.hh>
#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
#include <boost/numeric/odeint/stepper/bulirsch_stoer.hpp>
#include <gyronimo/dynamics/odeint_adapter.hh>
#include <gyronimo/core/codata.hh>
namespace py = pybind11;
using namespace gyronimo;

std::vector< std::vector<double>> gc_solver_Mercier(
  int field_periods, double Bref, double lambda_current,
  const std::vector<double>& s_grid,
  const std::vector<double>& curvature,
  const std::vector<double>& torsion,
  const std::vector<double>& B0,
  const std::vector<double>& phi2c,
  const std::vector<double>& phi2s,
  double charge, double rhom, double mass,
  double lambda, double energy, double r0,
  double theta0, double phi0, double nsamples,
  double Tfinal) {
  // Compute normalisation constants:
  double Valfven = Bref/std::sqrt(gyronimo::codata::mu0*(rhom*gyronimo::codata::m_proton*1.e+19));
  double Ualfven = 0.5*gyronimo::codata::m_proton*mass*Valfven*Valfven;
  double energySI = energy*gyronimo::codata::e;
  double vpp_sign = std::copysign(1.0, lambda);

  // Prepare metric, equilibrium and particles
  cubic_gsl_factory ifactory;

  metric_mercier g(field_periods, dblock_adapter(s_grid), dblock_adapter(curvature),
                   dblock_adapter(torsion), &ifactory);

  equilibrium_mercier qsc(&g, Bref, lambda_current, dblock_adapter(s_grid), dblock_adapter(B0),
            dblock_adapter(phi2c), dblock_adapter(phi2s), &ifactory);

  guiding_centre gc(1, Valfven, charge/mass, std::abs(lambda)*energySI/Ualfven/Bref, &qsc);
  guiding_centre::state initial_state = gc.generate_state(
      {r0, theta0, phi0}, energySI/Ualfven,(vpp_sign > 0 ? gyronimo::guiding_centre::plus : gyronimo::guiding_centre::minus));


  // Define variables for integration
  std::vector<std::vector< double >> x_vec;
  class push_back_state_and_time{
  public:
    std::vector< std::vector< double > >& m_states;
    push_back_state_and_time( std::vector< std::vector< double > > &states, 
      const IR3field_c1* e, const guiding_centre* g)
    : m_states( states ), eq_pointer_(e), gc_pointer_(g) { }
    void operator()(const guiding_centre::state& s, double t){
      IR3 x = gc_pointer_->get_position(s);
      double B = eq_pointer_->magnitude(x, t);
      guiding_centre::state dots = (*gc_pointer_)(s, t);
      IR3 y = gc_pointer_->get_position(dots);
      m_states.push_back({
        t,x[0],x[1],x[2],
        gc_pointer_->energy_parallel(s),
        gc_pointer_->energy_perpendicular(s, t),
        B, gc_pointer_->get_vpp(s), y[0], y[1], y[2],
        gc_pointer_->get_vpp(dots)
      });
    }
  private:
    const IR3field_c1* eq_pointer_;
    const guiding_centre* gc_pointer_;
  };

  // Integrate for t in [0,Tfinal], with dt=Tfinal/nsamples, using RK4.
  boost::numeric::odeint::bulirsch_stoer<guiding_centre::state> integration_algorithm;
//   boost::numeric::odeint::runge_kutta4<gyronimo::guiding_centre::state> integration_algorithm;
  boost::numeric::odeint::integrate_const(
      integration_algorithm, odeint_adapter(&gc),
      initial_state, 0.0, Tfinal, Tfinal/nsamples, push_back_state_and_time(x_vec,&qsc,&gc) );

  return x_vec;
}

// Python wrapper functions
PYBIND11_MODULE(NEAT_Mercier, m) {
    m.doc() = "Gyronimo Wrapper for the Stellarator Near-Axis Expansion (STELLNA) using the Mercier approach";
    m.def("gc_solver_Mercier",&gc_solver_Mercier);
}
