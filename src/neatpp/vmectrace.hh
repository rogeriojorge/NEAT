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

class push_back_state_and_time_vmec {
public:
  vector< vector< double > >& m_states;
  push_back_state_and_time_vmec(vector< vector< double > > &states,
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
        x[IR3::u], x[IR3::v], x[IR3::w],
        gc_pointer_->energy_parallel(s), 
        gc_pointer_->energy_perpendicular(s, t),
        B, v_parallel,
        y[IR3::u], y[IR3::v], y[IR3::w],
        gc_pointer_->get_vpp(dots),
        X[IR3::u], X[IR3::v], X[IR3::w]
      });
  };
private:
  const IR3field_c1* eq_pointer_;
  const guiding_centre* gc_pointer_;
};

vector< vector<double>>  vmectrace(
        string vmec_file,
        double charge, double mass, double Lambda,
        double vpp_sign, double energy, double s0,
        double theta0, double phi0,
        size_t nsamples, double Tfinal)
{
  parser_vmec vmap(vmec_file);
  cubic_gsl_factory ifactory;
  metric_vmec g(&vmap, &ifactory);
  equilibrium_vmec veq(&g, &ifactory);

  double Lref = 1.0;
  double Vref = 1.0;
  double refEnergy = 0.5*codata::m_proton*mass*Vref*Vref;
  double energySI = energy*codata::e;
  double energySI_over_refEnergy = energySI/refEnergy;
  double Bref = vmap.B_0();

  guiding_centre gc(
      Lref, Vref, charge/mass, Lambda*energySI_over_refEnergy, &veq);
  guiding_centre::state initial_state = gc.generate_state(
      {s0, theta0, phi0}, energySI_over_refEnergy,
      (vpp_sign > 0 ? guiding_centre::plus : guiding_centre::minus));

  cout.precision(16);
  cout.setf(ios::scientific);
  vector<vector< double >> x_vec;
  push_back_state_and_time_vmec observer(x_vec, &veq, &gc);
  boost::numeric::odeint::runge_kutta4<guiding_centre::state>
      integration_algorithm;
  boost::numeric::odeint::integrate_const(
      integration_algorithm, odeint_adapter(&gc),
      initial_state, 0.0, Tfinal, Tfinal/nsamples, observer);

  return x_vec;

}
