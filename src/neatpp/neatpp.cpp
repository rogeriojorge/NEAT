#include <cstddef>
#include <iterator>
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
#include <gyronimo/core/linspace.hh>
#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
#include <boost/numeric/odeint/stepper/bulirsch_stoer.hpp>
#include <gyronimo/dynamics/odeint_adapter.hh>
#include <gyronimo/core/codata.hh>
#include <numbers>
#include <omp.h>
#include <random>
#include <chrono>
// namespace py = pybind11;
using namespace gyronimo;

 std::vector< std::vector<double>> gc_solver_qs(
  double G0, double G2, double I2, double nfp, double iota,
  double iotaN, double B0, double B1c,
  double B20, double B2c, double beta1s, double charge,
  double mass, double lambda, int vpp_sign,
  double energy, double r0, double theta0, 
  double phi0, size_t nsamples, double Tfinal
 )
 {
  // Compute normalisation constants:
  double Lref = 1.0;
  double Vref = 1.0;
  double Uref = 0.5*gyronimo::codata::m_proton*mass*Vref*Vref;
  double energySI = energy*gyronimo::codata::e;

  // Prepare metric, equilibrium and particles
  double Bref = B0;
  metric_stellna_qs g(Bref, G0, G2, I2, iota, iotaN,
                     B0, B1c, B20, B2c, beta1s);

  equilibrium_stellna_qs qsc(&g);

  guiding_centre gc(Lref, Vref, charge/mass, lambda*energySI/Uref/Bref, &qsc);
  guiding_centre::state initial_state = gc.generate_state(
      {r0, theta0, phi0}, energySI/Uref,(vpp_sign > 0 ? gyronimo::guiding_centre::plus : gyronimo::guiding_centre::minus));

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

template<typename Gyron>
class ensemble {
// Gyron::state should be required to be a contiguous type!
public:
    typedef std::vector<typename Gyron::state> state;
    ensemble(const std::vector<Gyron>& gyron, const std::size_t n_particles_per_lambda) : gyron_(gyron), n_particles_per_lambda_(n_particles_per_lambda) {};
    void operator()(const state& f, state& dfdx, double t) const {
    #pragma omp parallel for
    for(std::size_t j = 0;j < n_particles_per_lambda_;j++){
        for(std::size_t k = 0;k < gyron_.size();k++){
                dfdx[j + k*n_particles_per_lambda_] = gyron_[k](f[j + k*n_particles_per_lambda_], t);
            }
        }
    };
    size_t size() const {return gyron_.size()*n_particles_per_lambda_;};
    size_t n_particles_per_lambda() const {return n_particles_per_lambda_;};
    const std::vector<Gyron>& gyron() const {return gyron_;};
private:
const std::vector<Gyron> gyron_;
const size_t n_particles_per_lambda_;
};

  std::vector< std::vector<double>> gc_solver_qs_ensemble(
  double G0, double G2, double I2, double nfp, double iota,
  double iotaN, double B0, double B1c, double B20, double B2c,
  double beta1s, double charge, double mass, double energy,
  size_t nlambda_trapped, size_t nlambda_passing, double r0, double r_max,
  size_t ntheta, size_t nphi, size_t nsamples, double Tfinal, size_t nthreads
 )
 {

// gets the number of threads from the openmp environment:
  omp_set_dynamic(0);  // explicitly disable dynamic teams
  omp_set_num_threads(nthreads);

     // defines the ensemble and dynamical system:
    typedef ensemble<gyronimo::guiding_centre> ensemble_type;

std::vector<std::vector< double >> x_vec;
// ODEInt observer object to print diagnostics at each time step.
class orbit_observer {
public:
  std::vector< std::vector< double > >& m_states;
  orbit_observer( std::vector< std::vector< double > > &states, const ensemble_type& particle_ensemble)
  : m_states( states ), particle_ensemble_(particle_ensemble) { };
  void operator()(const ensemble_type::state& z, double t) {
    std::vector<double> temp;
    temp.push_back(t);
    for(std::size_t k = 0;k < particle_ensemble_.size();k++){
      IR3 x = particle_ensemble_.gyron()[k/particle_ensemble_.n_particles_per_lambda()].get_position(z[k]);
      temp.push_back(x[0]);
    }
    m_states.push_back(temp);
  };
private:
 const ensemble_type& particle_ensemble_;
};

  double Bref = B0;
  metric_stellna_qs g(Bref, G0, G2, I2, iota, iotaN,
                     B0, B1c, B20, B2c, beta1s);

  equilibrium_stellna_qs qsc(&g);

  // Compute normalisation constants:
  double Lref = 1.0;
  double Vref = 1.0;
  double Uref = 0.5*gyronimo::codata::m_proton*mass*Vref*Vref;
  double energySI = energy*gyronimo::codata::e;
  double B_max = abs(B0) + abs(r_max * B1c) + r_max * r_max * (abs(B20) + abs(B2c));
  double B_min = std::max( 0.01, abs(B0) - abs(r_max * B1c) - r_max * r_max * (abs(B20) + abs(B2c)) );

// # As we work in Boozer coordinates, not in spacial coordinates, we don't initialize particles
// # uniformly in cartesian coordinates, in real space. To alleviate that, each particle initialization
// # or the objective function for each particle can be weighted by the volume jacobian
// # Jacobian in Boozer coordinates = (G/B^2)(r_0,theta_0,phi_0), ((G-N*I)/B^2)(r_0,theta_0,phi_0) if theta is theta-N phi (check!)        
  std::valarray<double> theta = linspace<std::valarray<double>>(0.0, 2*std::numbers::pi, ntheta);
  std::valarray<double> phi = linspace<std::valarray<double>>(0.0, 2*std::numbers::pi/nfp, nphi);
  std::valarray<double> lambda_trapped = linspace<std::valarray<double>>(B0/B_max, B0/B_min, nlambda_trapped);
  std::valarray<double> lambda_passing = linspace<std::valarray<double>>(0.0, B0/B_max*(1.0-1.0/nlambda_passing), nlambda_passing);
  // lambda minimo = 0?
  
  std::vector<guiding_centre> guiding_centre_vector;
  // de modo a poder paralelizar estes ciclos, dimensionar o array initial com o tamanho (nlambda_trapped+nlambda_passing)*ntheta*nphi
//   #pragma omp parallel for
  for(std::size_t k = 0;k < nlambda_trapped;k++){
    guiding_centre_vector.push_back(guiding_centre(Lref, Vref, charge/mass, lambda_trapped[k]*energySI/Uref/Bref, &qsc));
  }
  for(std::size_t k = 0;k < nlambda_passing;k++){
    guiding_centre_vector.push_back(guiding_centre(Lref, Vref, charge/mass, lambda_passing[k]*energySI/Uref/Bref, &qsc));
  }

  // defines the ensemble initial state:
  ensemble_type::state initial;
  // de modo a poder paralelizar estes ciclos, dimensionar o array initial com o tamanho (nlambda_trapped+nlambda_passing)
//   #pragma omp parallel for
    for(std::size_t k = 0;k < nlambda_trapped + nlambda_passing;k++){
        for(std::size_t j = 0;j < ntheta;j++){
            for(std::size_t l = 0;l < nphi;l++){
                initial.push_back(guiding_centre_vector[k].generate_state(
                    {r0, theta[j], phi[l]}, energySI/Uref,
                    gyronimo::guiding_centre::vpp_sign::plus));
                initial.push_back(guiding_centre_vector[k].generate_state(
                    {r0, theta[j], phi[l]}, energySI/Uref,
                    gyronimo::guiding_centre::vpp_sign::minus));
            }
        }
    }

// create ensemble object
ensemble_type ensemble_object(guiding_centre_vector, 2 * ntheta * nphi);

// integrates for t in [0,Tfinal], with dt=Tfinal/nsamples, using RK4.
  boost::numeric::odeint::runge_kutta4<ensemble_type::state> ode_stepper;
  boost::numeric::odeint::integrate_const(
      ode_stepper, ensemble_object,
      initial, 0.0, Tfinal, Tfinal/nsamples, orbit_observer(x_vec, ensemble_object)
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
