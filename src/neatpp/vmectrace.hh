#ifdef OPENMP
#include <omp.h>
#endif
#include <cmath>
#include <argh.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gyronimo/version.hh>
#include <gyronimo/core/codata.hh>
#include <gyronimo/core/linspace.hh>
#include <gyronimo/parsers/parser_vmec.hh>
#include <gyronimo/fields/equilibrium_vmec.hh>
#include <gyronimo/interpolators/cubic_gsl.hh>
#include <gyronimo/dynamics/guiding_centre.hh>
#include <gyronimo/dynamics/odeint_adapter.hh>
#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>
using namespace gyronimo;
using namespace std;

class orbit_observer_new {
public:
  vector< vector< double > >& m_states;
  orbit_observer_new(vector< vector< double > > &states,
                     const IR3field_c1* e, const guiding_centre* g)
    : m_states(states), eq_pointer_(e), gc_pointer_(g) {};
  void operator()(const guiding_centre::state& s, double t) {
    IR3 x = gc_pointer_->get_position(s);
    double B = eq_pointer_->magnitude(x, t);
    guiding_centre::state dots = (*gc_pointer_)(s, t);
    IR3 y = gc_pointer_->get_position(dots);
    IR3 X = eq_pointer_->metric()->transform2cylindrical(x);
    double v_parallel = gc_pointer_->get_vpp(s);
    m_states.push_back({
        t,
        // x[IR3::u], x[IR3::v], x[IR3::w],
        X[IR3::u], X[IR3::v], X[IR3::w],
        gc_pointer_->energy_parallel(s), 
        gc_pointer_->energy_perpendicular(s, t),
        B, v_parallel,
        y[IR3::u], y[IR3::v], y[IR3::w],
        gc_pointer_->get_vpp(dots)
      });
  };
private:
  const IR3field_c1* eq_pointer_;
  const guiding_centre* gc_pointer_;
};

vector< vector<double>>  vmectrace(
        double mass,double charge,double energy,double s0,
        double theta0,double phi0,double Lambda,double Tfinal,
        size_t nsamples, string vmec_file)
{
  parser_vmec vmap(vmec_file);
  cubic_gsl_factory ifactory;
  metric_vmec g(&vmap, &ifactory);
  equilibrium_vmec veq(&g, &ifactory);
  
  double vpp_sign = copysign(1.0, Lambda);  // Lambda carries vpp sign.
  Lambda = abs(Lambda);  // once vpp sign is stored, Lambda turns unsigned.

// Computes normalisation constants:
  double Vref = 1; // New version of Valfven
  double Uref = 0.5*codata::m_proton*mass*Vref*Vref; // New version of Ualfven
  double energySI = energy*codata::e;
  double Lref= 1.0;

// Builds the guiding_centre object:
  guiding_centre gc(
      Lref, Vref, charge/mass, Lambda*energySI/Uref, &veq);

// Computes the initial conditions from the supplied constants of motion:
  /*double zstar = charge*g.parser()->cpsurf()*veq.B_0()*veq.R_0()*veq.R_0();
  double vstar = Vref*mass*codata::m_proton/codata::e;*/
  //double vdagger = vstar*sqrt(energySI/Uref);
  guiding_centre::state initial_state = gc.generate_state(
      {s0, theta0, phi0}, energySI/Uref,
      (vpp_sign > 0 ?
        guiding_centre::plus : guiding_centre::minus));

// integrates for t in [0,Tfinal], with dt=Tfinal/nsamples, using RK4.
  cout.precision(16);
  cout.setf(ios::scientific);
  vector<vector< double >> x_vec;
  orbit_observer_new observer(x_vec, &veq, &gc);

  boost::numeric::odeint::runge_kutta4<guiding_centre::state>
      integration_algorithm;
  boost::numeric::odeint::integrate_const(
      integration_algorithm, odeint_adapter(&gc),
      initial_state, 0.0, Tfinal, Tfinal/nsamples, observer);

  return x_vec;

}
