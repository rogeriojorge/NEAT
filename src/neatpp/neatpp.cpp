#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// // // //  USE CUBIC_GSL_PERIODIC
#include <gyronimo/interpolators/cubic_gsl.hh>
#include <gyronimo/interpolators/steffen_gsl.hh>
#include "metric_stellna_qs.hh"
#include <gyronimo/core/dblock.hh>
#include <vector>
#include "equilibrium_stellna_qs.hh"
#include <gyronimo/dynamics/guiding_centre.hh>
#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
#include <boost/numeric/odeint/stepper/bulirsch_stoer.hpp>
#include <gyronimo/dynamics/odeint_adapter.hh>
#include <gyronimo/core/codata.hh>
#include <omp.h>
#include <random>
#include <chrono>
// namespace py = pybind11;
using namespace gyronimo;

 std::vector< std::vector<double>> gc_solver_qs(
  double G0, double G2, double I2, double iota,
  double iotaN, double Bref, double B0, double B1c,
  double B20, double B2c, double beta1s, double charge,
  double rhom, double mass, double lambda,
  double energy, double r0, double theta0, 
  double phi0, double nsamples, double Tfinal
 )
 {
  // Compute normalisation constants:
  double Valfven = Bref/std::sqrt(gyronimo::codata::mu0*(rhom*gyronimo::codata::m_proton*1.e+19));
  double Ualfven = 0.5*gyronimo::codata::m_proton*mass*Valfven*Valfven;
  double energySI = energy*gyronimo::codata::e;
  double vpp_sign = std::copysign(1.0, lambda);

  // Prepare metric, equilibrium and particles
  metric_stellna_qs g(Bref, G0, G2, I2, iota, iotaN,
                     B0, B1c, B20, B2c, beta1s);

  equilibrium_stellna_qs qsc(&g);

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
//   boost::numeric::odeint::runge_kutta4<gyronimo::guiding_centre::state> integration_algorithm;
  boost::numeric::odeint::bulirsch_stoer<gyronimo::guiding_centre::state> integration_algorithm;
  boost::numeric::odeint::integrate_const(
      integration_algorithm, odeint_adapter(&gc),
      initial_state, 0.0, Tfinal, Tfinal/nsamples, push_back_state_and_time(x_vec,&qsc,&gc) );

  return x_vec;
 }

std::array<double,4> operator*(const double& a, const std::array<double,4>& v) {
std::array<double, 4> result = {a*v[0], a*v[1], a*v[2], a*v[3]};
return result;
}
std::array<double,4> operator+(
    const std::array<double,4>& u, const std::array<double,4>& v) {
std::array<double, 4> result = {u[0]+v[0],u[1]+v[1],u[2]+v[2],u[3]+v[3]};
return result;
}

template<typename Gyron, size_t Size>
class ensemble {
// Gyron::state should be required to be a contiguous type!
public:
static const size_t size = Size;
typedef std::array<typename Gyron::state, Size> state;
ensemble(const Gyron* gyron) : gyron_(gyron) {};
void operator()(const state& f, state& dfdx, double t) const {
#pragma omp parallel for
    for(std::size_t k = 0;k < Size;k++)
    dfdx[k] = (*gyron_)(f[k], t);
};
private:
const Gyron* gyron_;
};

  std::vector< std::vector<double>> gc_solver_qs_ensemble(
  double G0, double G2, double I2, double iota,
  double iotaN, double Bref, double B0, double B1c,
  double B20, double B2c, double beta1s, double charge,
  double rhom, double mass, double energy, double r0, double theta0,
  double phi0, double vparallel_min, double vparallel_max,
  int nparticles,  double nsamples, double Tfinal, int nthreads
 )
 {
     // defines the ensemble size and dynamical system:
     const int _nparticles = 10000;
    typedef ensemble<gyronimo::guiding_centre, _nparticles> ensemble_type;

std::vector<std::vector< double >> x_vec;
// ODEInt observer object to print diagnostics at each time step.
class orbit_observer {
public:
  std::vector< std::vector< double > >& m_states;
  orbit_observer( std::vector< std::vector< double > > &states, 
      const IR3field_c1* e, const guiding_centre* g)
  : m_states( states ), eq_pointer_(e), gc_pointer_(g) { };
  void operator()(const ensemble_type::state& z, double t) {
    std::vector<double> temp;
    temp.push_back(t);
    for(std::size_t k = 0;k < ensemble_type::size;k++){
      IR3 x = gc_pointer_->get_position(z[k]);
      temp.push_back(x[0]);
    }
    m_states.push_back(temp);
  };
private:
  const IR3field_c1* eq_pointer_;
  const guiding_centre* gc_pointer_;
};

  metric_stellna_qs g(Bref, G0, G2, I2, iota, iotaN,
                     B0, B1c, B20, B2c, beta1s);

  equilibrium_stellna_qs qsc(&g);

  // Compute normalisation constants:
  double Valfven = Bref/std::sqrt(gyronimo::codata::mu0*(rhom*gyronimo::codata::m_proton*1.e+19));
  double Ualfven = 0.5*gyronimo::codata::m_proton*mass*Valfven*Valfven;
  double energySI = energy*gyronimo::codata::e;

  double lambda = 0;
  guiding_centre gc(1, Valfven, charge/mass, std::abs(lambda)*energySI/Ualfven/Bref, &qsc);

// gets the number of threads from the openmp environment:
  omp_set_dynamic(0);  // explicitly disable dynamic teams
  omp_set_num_threads(nthreads);

// defines the ensemble initial state:
  ensemble_type::state initial;
  std::mt19937 rand_generator;
  std::uniform_real_distribution<> vpp_distro(vparallel_min, vparallel_max);
#pragma omp parallel for
  for(std::size_t k = 0;k < ensemble_type::size;k++)
    initial[k] = gc.generate_state(
        {r0, 0.0, 0.0}, vpp_distro(rand_generator),
        gyronimo::guiding_centre::vpp_sign::plus);

// integrates for t in [0,Tfinal], with dt=Tfinal/nsamples, using RK4.
  boost::numeric::odeint::runge_kutta4<ensemble_type::state> ode_stepper;
  boost::numeric::odeint::integrate_const(
      ode_stepper, ensemble_type(&gc),
      initial, 0.0, Tfinal, Tfinal/nsamples, orbit_observer(x_vec,&qsc,&gc)
      );

     return x_vec;
 }

// std::vector< std::vector<double>> gc_solver(
//   int field_periods,
//   double G0, double G2, double I2,
//   double iota, double iotaN, double Bref,
//   const std::vector<double>& phi_grid,
//   const std::vector<double>& B0,
//   const std::vector<double>& B1c,
//   const std::vector<double>& B1s,
//   const std::vector<double>& B20,
//   const std::vector<double>& B2c,
//   const std::vector<double>& B2s,
//   const std::vector<double>& beta0,
//   const std::vector<double>& beta1c,
//   const std::vector<double>& beta1s,
//   double charge, double rhom, double mass,
//   double lambda, double energy, double r0,
//   double theta0, double phi0, double nsamples,
//   double Tfinal) {
//   // Compute normalisation constants:
//   double Valfven = Bref/std::sqrt(gyronimo::codata::mu0*(rhom*gyronimo::codata::m_proton*1.e+19));
//   double Ualfven = 0.5*gyronimo::codata::m_proton*mass*Valfven*Valfven;
//   double energySI = energy*gyronimo::codata::e;
//   double vpp_sign = std::copysign(1.0, lambda);

//   // Prepare metric, equilibrium and particles
// //   cubic_gsl_factory ifactory;
//   steffen_gsl_factory ifactory;
//   metric_stellna g(field_periods, Bref, dblock_adapter(phi_grid), G0, G2, I2, iota, iotaN,
//                    dblock_adapter(B0), dblock_adapter(B1c), dblock_adapter(B1s),
//                    dblock_adapter(B20), dblock_adapter(B2c), dblock_adapter(B2s),
//                    dblock_adapter(beta0), dblock_adapter(beta1c), dblock_adapter(beta1s),
//                    &ifactory);

//   equilibrium_stellna qsc(&g);

//   guiding_centre gc(1, Valfven, charge/mass, std::abs(lambda)*energySI/Ualfven/Bref, &qsc);
//   guiding_centre::state initial_state = gc.generate_state(
//       {r0, theta0, phi0}, energySI/Ualfven,(vpp_sign > 0 ? gyronimo::guiding_centre::plus : gyronimo::guiding_centre::minus));

//   // Define variables for integration
//   std::vector<std::vector< double >> x_vec;
//   class push_back_state_and_time{
//   public:
//     std::vector< std::vector< double > >& m_states;
//     push_back_state_and_time( std::vector< std::vector< double > > &states, 
//       const IR3field_c1* e, const guiding_centre* g)
//     : m_states( states ), eq_pointer_(e), gc_pointer_(g) { }
//     void operator()(const guiding_centre::state& s, double t){
//       IR3 x = gc_pointer_->get_position(s);
//       double B = eq_pointer_->magnitude(x, t);
//       guiding_centre::state dots = (*gc_pointer_)(s, t);
//       IR3 y = gc_pointer_->get_position(dots);
//       m_states.push_back({
//         t,x[0],x[1],x[2],
//         gc_pointer_->energy_parallel(s),
//         gc_pointer_->energy_perpendicular(s, t),
//         B, gc_pointer_->get_vpp(s), y[0], y[1], y[2],
//         gc_pointer_->get_vpp(dots)
//       });
//     }
//   private:
//     const IR3field_c1* eq_pointer_;
//     const guiding_centre* gc_pointer_;
//   };

//   // Integrate for t in [0,Tfinal], with dt=Tfinal/nsamples, using RK4.
// //   boost::numeric::odeint::runge_kutta4<gyronimo::guiding_centre::state> integration_algorithm;
//   boost::numeric::odeint::bulirsch_stoer<gyronimo::guiding_centre::state> integration_algorithm;
//   boost::numeric::odeint::integrate_const(
//       integration_algorithm, odeint_adapter(&gc),
//       initial_state, 0.0, Tfinal, Tfinal/nsamples, push_back_state_and_time(x_vec,&qsc,&gc) );

//   return x_vec;
// }

// Python wrapper functions
PYBIND11_MODULE(neatpp, m) {
    m.doc() = "Gyronimo Wrapper for the Stellarator Near-Axis Expansion (STELLNA)";
    // m.def("gc_solver",&gc_solver);
    m.def("gc_solver_qs",&gc_solver_qs);
    m.def("gc_solver_qs_ensemble",&gc_solver_qs_ensemble);
}
