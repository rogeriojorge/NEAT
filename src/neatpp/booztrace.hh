#ifdef OPENMP
#include <omp.h>
#endif
#include <chrono>
#include <cmath>
#include <argh.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gyronimo/version.hh>
#include <gyronimo/core/codata.hh>
#include <gyronimo/core/linspace.hh>
#include "parser_boozxform.hh"
#include "equilibrium_boozxform.hh"
#include "metric_boozxform.hh"
#include <gyronimo/interpolators/cubic_gsl.hh>
#include <gyronimo/dynamics/guiding_centre.hh>
#include <gyronimo/dynamics/odeint_adapter.hh>
#include <boost/math/tools/roots.hpp>

#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_cash_karp54.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_fehlberg78.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>
#include <boost/numeric/odeint/stepper/adams_bashforth.hpp>
#include <boost/numeric/odeint/stepper/adams_bashforth_moulton.hpp>
#include <boost/numeric/odeint/stepper/adams_moulton.hpp>
#include <boost/numeric/odeint/stepper/bulirsch_stoer.hpp>

#include <boost/numeric/odeint/integrate/integrate_const.hpp>
#include <boost/numeric/odeint/integrate/integrate_adaptive.hpp>
#include <gsl/gsl_errno.h>
using namespace gyronimo;
using namespace std;
using namespace boost::numeric::odeint;


class push_back_state_and_time_boozxform
{
public:
  vector<vector<double>> &m_states;
  double &maximum_s_;
  int &integrator_;
  push_back_state_and_time_boozxform(vector<vector<double>> &states, int &integrator,
                                const IR3field_c1 *e, const guiding_centre *g, double &maximum_s)
      : m_states(states), integrator_(integrator), eq_pointer_(e), gc_pointer_(g), maximum_s_(maximum_s){};
  void operator()(const guiding_centre::state s, double t)
  {
    IR3 x = gc_pointer_->get_position(s);
    double B = (eq_pointer_->magnitude(x, t)) * eq_pointer_->m_factor();
    auto start_time = std::chrono::high_resolution_clock::now();
    guiding_centre::state dots = (*gc_pointer_)(s, t);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    
    IR3 y = gc_pointer_->get_position(dots);
    IR3 X = eq_pointer_->metric()->transform2cylindrical(x);
    double v_parallel = gc_pointer_->get_vpp(s);
    IR3 B_cov = (eq_pointer_->covariant(x, t)) * eq_pointer_->m_factor();
    IR3 B_contrav = (eq_pointer_->contravariant(x, t)) * eq_pointer_->m_factor();
    double jac = eq_pointer_->metric()->jacobian(x);
    std::cout << "BOOZTRACE" << std::endl;
    std::cout << "jac: " << jac << std::endl;
    std::cout << "B: " << B << std::endl;
    std::cout << "B_cov[IR3::u]:" << B_cov[IR3::u] << std::endl;
    std::cout << "B_cov[IR3::v]:" << B_cov[IR3::v] << std::endl;
    std::cout << "B_cov[IR3::w]:" << B_cov[IR3::w] << std::endl;
    std::cout << "B_contrav[IR3::u]:" << B_contrav[IR3::u] << std::endl;
    std::cout << "B_contrav[IR3::v]:" << B_contrav[IR3::v] << std::endl;
    std::cout << "B_contrav[IR3::w]:" << B_contrav[IR3::w] << std::endl;
    // exit(0);
    double minimum_s_ = 0.1;

    m_states.push_back({t,
                        x[IR3::u], x[IR3::v], x[IR3::w],
                        gc_pointer_->energy_parallel(s),
                        gc_pointer_->energy_perpendicular(s, t),
                        B, v_parallel,
                        y[IR3::u], y[IR3::v], y[IR3::w],
                        gc_pointer_->get_vpp(dots),
                        X[IR3::u], X[IR3::v], X[IR3::w],
                        B_cov[IR3::u], B_cov[IR3::v], B_cov[IR3::w],
                        B_contrav[IR3::u], B_contrav[IR3::v], B_contrav[IR3::w]});
  };

private:
  const IR3field_c1 *eq_pointer_;
  const guiding_centre *gc_pointer_;
};

class cached_metric_trace_booz : public gyronimo::metric_boozxform {
 public:
  cached_metric_trace_booz(
      const gyronimo::parser_boozxform* p, const gyronimo::interpolator1d_factory* f)
      : metric_boozxform(p, f) {};
  virtual gyronimo::SM3 operator()(const gyronimo::IR3& x) const override {
    thread_local gyronimo::IR3 cx = {0,0,0};
    thread_local gyronimo::SM3 cg = {0,0,0,0,0,0};
    if(boost::hash_value(x) != boost::hash_value(cx)) {
      cg = gyronimo::metric_boozxform::operator()(x);
      cx = x;
    }
    return cg;
  };
  virtual gyronimo::dSM3 del(const gyronimo::IR3& x) const override {
    thread_local gyronimo::IR3 cx = {0,0,0};
    thread_local gyronimo::dSM3 cdg = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    if(boost::hash_value(x) != boost::hash_value(cx)) {
      cdg = gyronimo::metric_boozxform::del(x);
      cx = x;
    }
    return cdg;
  };
};

class cached_field_trace_booz : public gyronimo::equilibrium_boozxform {
 public:
  cached_field_trace_booz(
      const gyronimo::metric_boozxform* p, const gyronimo::interpolator1d_factory* f)
      : equilibrium_boozxform(p, f) {};
  virtual gyronimo::IR3 contravariant(
      const gyronimo::IR3& x, double t) const override {
    thread_local double ct = -1;
    thread_local gyronimo::IR3 cx = {0,0,0};
    thread_local gyronimo::IR3 cg = {0,0,0};
    if(boost::hash_value(x) != boost::hash_value(cx) || t != ct) {
      cg = gyronimo::equilibrium_boozxform::contravariant(x, t);
      cx = x;
      ct = t;
    }
    return cg;
  };
  virtual gyronimo::dIR3 del_contravariant(
      const gyronimo::IR3& x, double t) const override {
    thread_local double ct = -1;
    thread_local gyronimo::IR3 cx = {0,0,0};
    thread_local gyronimo::dIR3 cdg = {0,0,0,0,0,0,0,0,0};
    if(boost::hash_value(x) != boost::hash_value(cx) || t != ct) {
      cdg = gyronimo::equilibrium_boozxform::del_contravariant(x, t);
      cx = x;
      ct = t;
    }
    return cdg;
  };
};

vector<vector<double>> booztrace(
    string boozxform_file, double maximum_s, int integrator,
    double charge, double mass, double Lambda,
    double vpp_sign, double energy, double s0,
    double theta0, double phi0,
    size_t nsamples, double Tfinal)
{
  parser_boozxform vmap(boozxform_file);
  cubic_gsl_factory ifactory;
  // metric_boozxform g(&vmap, &ifactory);
  cached_metric_trace_booz g(&vmap, &ifactory);
  // equilibrium_boozxform veq(&g, &ifactory);
  cached_field_trace_booz veq(&g, &ifactory);
  
  double Lref = 1.0;
  double Vref = 1.0;
  double refEnergy = 0.5 * codata::m_proton * mass * Vref * Vref;
  double energySI = energy * codata::e;
  double energySI_over_refEnergy = energySI / refEnergy;
  double Bi = veq.magnitude({s0, phi0, theta0}, 0);
  // double Bi = veq.B_0();
//   cout << "Booz: " << Bi << endl;

  guiding_centre gc(Lref, Vref, charge / mass, Lambda * energySI_over_refEnergy / Bi , &veq); // -> Version with Bi
  guiding_centre::state initial_state = gc.generate_state(
      {s0, theta0, phi0}, energySI_over_refEnergy,
      (vpp_sign > 0 ? guiding_centre::plus : guiding_centre::minus));

  cout.precision(16);
  cout.setf(ios::scientific);
  vector<vector<double>> x_vec;
  push_back_state_and_time_boozxform observer(x_vec, integrator, &veq, &gc, maximum_s);
  gsl_set_error_handler_off();

 
  runge_kutta_dopri5<guiding_centre::state> integration_algorithm; //runge_kutta_cash_karp54<guiding_centre::state> integration_algorithm;
  try
  {
    integrate_const(integration_algorithm, odeint_adapter(&gc),
        initial_state, 0.0, Tfinal, Tfinal / nsamples, observer);
  }
  catch (int e)
  { } 

  return x_vec;
}
