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
#include <gyronimo/parsers/parser_vmec.hh>
#include <gyronimo/fields/equilibrium_vmec.hh>
#include <gyronimo/interpolators/cubic_gsl.hh>
#include <gyronimo/dynamics/guiding_centre.hh>
#include <gyronimo/dynamics/odeint_adapter.hh>
#include <boost/math/tools/roots.hpp>
// Integrators
#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_cash_karp54.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_fehlberg78.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>
#include <boost/numeric/odeint/stepper/adams_bashforth.hpp>
#include <boost/numeric/odeint/stepper/adams_bashforth_moulton.hpp>
#include <boost/numeric/odeint/stepper/adams_moulton.hpp>
#include <boost/numeric/odeint/stepper/bulirsch_stoer.hpp>
// Const and adaptive integration support
#include <boost/numeric/odeint/integrate/integrate_const.hpp>
#include <boost/numeric/odeint/integrate/integrate_adaptive.hpp>
#include <gsl/gsl_errno.h>
using namespace gyronimo;
using namespace std;
using namespace boost::numeric::odeint;


class push_back_state_and_time_vmec
{
public:
  vector<vector<double>> &m_states;
  double &maximum_s_;
  int &integrator_;
  push_back_state_and_time_vmec(vector<vector<double>> &states, int &integrator,
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
    // Benchmarking of co and contravariant B
    // IR3 B_cov = (eq_pointer_->covariant(x, t)) * eq_pointer_->m_factor();
    // IR3 B_con = (eq_pointer_->contravariant(x, t)) * eq_pointer_->m_factor();

    m_states.push_back({t,
                        x[IR3::u], x[IR3::w], x[IR3::v],
                        gc_pointer_->energy_parallel(s),
                        gc_pointer_->energy_perpendicular(s, t),
                        B, v_parallel,
                        y[IR3::u], y[IR3::w], y[IR3::v],
                        gc_pointer_->get_vpp(dots),
                        X[IR3::u], X[IR3::v], X[IR3::w],
                        // B_cov[IR3::u], B_cov[IR3::w], B_cov[IR3::v],
                        // B_con[IR3::u], B_con[IR3::w], B_con[IR3::v]
                      });
  };

private:
  const IR3field_c1 *eq_pointer_;
  const guiding_centre *gc_pointer_;
};

class cached_metric_trace : public gyronimo::metric_vmec {
 public:
  cached_metric_trace(
      const gyronimo::parser_vmec* p, const gyronimo::interpolator1d_factory* f)
      : metric_vmec(p, f) {};
  virtual gyronimo::SM3 operator()(const gyronimo::IR3& x) const override {
    thread_local gyronimo::IR3 cx = {0,0,0};
    thread_local gyronimo::SM3 cg = {0,0,0,0,0,0};
    if(boost::hash_value(x) != boost::hash_value(cx)) {
      cg = gyronimo::metric_vmec::operator()(x);
      cx = x;
    }
    return cg;
  };
  virtual gyronimo::dSM3 del(const gyronimo::IR3& x) const override {
    thread_local gyronimo::IR3 cx = {0,0,0};
    thread_local gyronimo::dSM3 cdg = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    if(boost::hash_value(x) != boost::hash_value(cx)) {
      cdg = gyronimo::metric_vmec::del(x);
      cx = x;
    }
    return cdg;
  };
};

class cached_field_trace : public gyronimo::equilibrium_vmec {
 public:
  cached_field_trace(
      const gyronimo::metric_vmec* p, const gyronimo::interpolator1d_factory* f)
      : equilibrium_vmec(p, f) {};
  virtual gyronimo::IR3 contravariant(
      const gyronimo::IR3& x, double t) const override {
    thread_local double ct = -1;
    thread_local gyronimo::IR3 cx = {0,0,0};
    thread_local gyronimo::IR3 cg = {0,0,0};
    if(boost::hash_value(x) != boost::hash_value(cx) || t != ct) {
      cg = gyronimo::equilibrium_vmec::contravariant(x, t);
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
      cdg = gyronimo::equilibrium_vmec::del_contravariant(x, t);
      cx = x;
      ct = t;
    }
    return cdg;
  };
};

vector<vector<double>> vmectrace(
    string vmec_file, double maximum_s, int integrator,
    double charge, double mass, double Lambda,
    double vpp_sign, double energy, double s0,
    double theta0, double phi0,
    size_t nsamples, double Tfinal)
{
  parser_vmec vmap(vmec_file);
  cubic_gsl_factory ifactory;
  // Cached metrics and fields - faster but memory intensive
  cached_metric_trace g(&vmap, &ifactory);
  cached_field_trace veq(&g, &ifactory);
  // No cache version - slower but less memory
  // metric_vmec g(&vmap, &ifactory);
  // equilibrium_vmec veq(&g, &ifactory);
  
  double Lref = 1.0;
  double Vref = 1.0;
  double refEnergy = 0.5 * codata::m_proton * mass * Vref * Vref;
  double energySI = energy * codata::e;
  double energySI_over_refEnergy = energySI / refEnergy;
  double Bi = veq.magnitude({s0, phi0, theta0}, 0);

  // Lambda*energySI_over_refEnergy = energy*Bref/(2*Binicial*Uref)*(1-vparallel_over_v^2)

  guiding_centre gc(Lref, Vref, charge / mass, Lambda * energySI_over_refEnergy / Bi , &veq);
  guiding_centre::state initial_state = gc.generate_state(
      {s0, phi0, theta0}, energySI_over_refEnergy,
      (vpp_sign > 0 ? guiding_centre::plus : guiding_centre::minus));

  cout.precision(16);
  cout.setf(ios::scientific);
  vector<vector<double>> x_vec;
  push_back_state_and_time_vmec observer(x_vec, integrator, &veq, &gc, maximum_s);
  
  gsl_set_error_handler_off();

// Constant integration comparisons
  switch (integrator) {
    case 1:
      { 
      runge_kutta4<guiding_centre::state> integration_algorithm;
      try
      {
        integrate_const(integration_algorithm, odeint_adapter(&gc),
            initial_state, 0.0, Tfinal, Tfinal / nsamples, observer);
      }
      catch (int e)
      { } 
      break;}
    case 2:
      {   
      runge_kutta_cash_karp54<guiding_centre::state> integration_algorithm; //runge_kutta_cash_karp54<guiding_centre::state> integration_algorithm;
      try
      {
        integrate_const(integration_algorithm, odeint_adapter(&gc),
            initial_state, 0.0, Tfinal, Tfinal / nsamples, observer);
      }
      catch (int e)
      { } 
      break;}
    case 3:
      {   
      runge_kutta_fehlberg78<guiding_centre::state> integration_algorithm;
      try
      {
        integrate_const(integration_algorithm, odeint_adapter(&gc),
            initial_state, 0.0, Tfinal, Tfinal / nsamples, observer);
      }
      catch (int e)
      { } 
      break;}
    case 4:
      {   
      runge_kutta_dopri5<guiding_centre::state> integration_algorithm;
      try
      {
        integrate_const(integration_algorithm, odeint_adapter(&gc),
            initial_state, 0.0, Tfinal, Tfinal / nsamples, observer);
      }
      catch (int e)
      { } 
      break;}
    case 5:
      {   
      bulirsch_stoer<guiding_centre::state> integration_algorithm;
      try
      {
        integrate_const(integration_algorithm, odeint_adapter(&gc),
            initial_state, 0.0, Tfinal, Tfinal / nsamples, observer);
      }
      catch (int e)
      { } 
      break;}
    case 6:
      {   
      adams_bashforth<5,guiding_centre::state> integration_algorithm;
      try
      {
        integrate_const(integration_algorithm, odeint_adapter(&gc),
            initial_state, 0.0, Tfinal, Tfinal / nsamples, observer);
      }
      catch (int e)
      { } 
      break;}
    case 7:
      {   
      adams_bashforth_moulton<5,guiding_centre::state> integration_algorithm;
      try
      {
        integrate_const(integration_algorithm, odeint_adapter(&gc),
            initial_state, 0.0, Tfinal, Tfinal / nsamples, observer);
      }
      catch (int e)
      { } 
      break;}
    case 8:
      {   
      adams_bashforth<8,guiding_centre::state> integration_algorithm;
      try
      {
        integrate_const(integration_algorithm, odeint_adapter(&gc),
            initial_state, 0.0, Tfinal, Tfinal / nsamples, observer);
      }
      catch (int e)
      { } 
      break;}
    case 9:
      {   
      adams_bashforth_moulton<8,guiding_centre::state> integration_algorithm;
      try
      {
        integrate_const(integration_algorithm, odeint_adapter(&gc),
            initial_state, 0.0, Tfinal, Tfinal / nsamples, observer);
      }
      catch (int e)
      { } 
      break;}
    default:
      cout << "Error: invalid stepper choice." << endl;
      break;
  }

  // Const and adaptive integrators
  // switch (integrator) {
  //   case 1:
  //     {   
  //     runge_kutta_cash_karp54<guiding_centre::state> integration_algorithm;
  //     try
  //     {
  //       integrate_const(integration_algorithm, odeint_adapter(&gc),
  //           initial_state, 0.0, Tfinal, Tfinal / nsamples, observer);
  //     }
  //     catch (int e)
  //     { } 
  //     break;}
  //   case 2:
  //     {   
  //     runge_kutta_cash_karp54<guiding_centre::state> integration_algorithm;
  //     typedef runge_kutta_cash_karp54<guiding_centre::state> error_stepper_type;
  //     try
  //     {
  //       integrate_adaptive(make_controlled( (1.0e-14)*(100/nsamples) , (1.0e-16) , error_stepper_type() ), odeint_adapter(&gc),
  //         initial_state, 0.0, Tfinal, Tfinal/nsamples, observer);
  //     }
  //     catch (int e)
  //     { } 
  //     break;}
  //   case 3:
  //     {   
  //     runge_kutta_fehlberg78<guiding_centre::state> integration_algorithm;
  //     try
  //     {
  //       integrate_const(integration_algorithm, odeint_adapter(&gc),
  //           initial_state, 0.0, Tfinal, Tfinal / nsamples, observer);
  //     }
  //     catch (int e)
  //     { } 
  //     break;}
  //   case 4:
  //     {   
  //     runge_kutta_fehlberg78<guiding_centre::state> integration_algorithm;
  //     typedef runge_kutta_fehlberg78<guiding_centre::state> error_stepper_type;
  //     try
  //     {
  //       integrate_adaptive(make_controlled( (1.0e-14)*(100/nsamples) , (1.0e-16) , error_stepper_type() ), odeint_adapter(&gc),
  //         initial_state, 0.0, Tfinal, Tfinal/nsamples, observer);
  //     }
  //     catch (int e)
  //     { } 
  //     break;}
  //   case 5:
  //     {   
  //     runge_kutta_dopri5<guiding_centre::state> integration_algorithm;
  //     try
  //     {
  //       integrate_const(integration_algorithm, odeint_adapter(&gc),
  //           initial_state, 0.0, Tfinal, Tfinal / nsamples, observer);
  //     }
  //     catch (int e)
  //     { } 
  //     break;}
  //   case 6:
  //     {   
  //     runge_kutta_dopri5<guiding_centre::state> integration_algorithm;
  //     typedef runge_kutta_dopri5<guiding_centre::state> error_stepper_type;
  //     try
  //     {
  //       integrate_adaptive(make_controlled( (1.0e-14)*(100/nsamples) , (1.0e-16) , error_stepper_type() ), odeint_adapter(&gc),
  //         initial_state, 0.0, Tfinal, Tfinal/nsamples, observer);
  //     }
  //     catch (int e)
  //     { } 
  //     break;}
  //   case 7:
  //     {   
  //     bulirsch_stoer<guiding_centre::state> integration_algorithm;
  //     try
  //     {
  //       integrate_const(integration_algorithm, odeint_adapter(&gc),
  //           initial_state, 0.0, Tfinal, Tfinal / nsamples, observer);
  //     }
  //     catch (int e)
  //     { } 
  //     break;}
  //   // case 8:
  //   //   {   
  //   //   bulirsch_stoer<guiding_centre::state> integration_algorithm;
  //   //   typedef bulirsch_stoer<guiding_centre::state> error_stepper_type;
  //   //   try
  //   //   {
  //   //     integrate_adaptive(make_controlled( (1.0e-14)*(100/nsamples) , (1.0e-16) , error_stepper_type() ), odeint_adapter(&gc),
  //   //       initial_state, 0.0, Tfinal, Tfinal/nsamples, observer);
  //   //   }
  //   //   catch (int e)
  //   //   { } 
  //   //   break;}
  //   default:
  //     cout << "Error: invalid stepper choice." << endl;
  //     break;
  // }
  
  // boost::numeric::odeint::integrate_adaptive(
  //   integration_algorithm, odeint_adapter(&gc),
  //   initial_state, 0.0, Tfinal, Tfinal/nsamples, observer);
  return x_vec;
}
