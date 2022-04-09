// // // NEAT: NEar-Axis sTellarator Particle Tracer
// This file is a python Wrapper
// It allows gyronimo to be called directly from Python
// Rogerio Jorge, July 2021, Greifswald
// // //
// Command to compile NEAT on a Macbook with gsl pre-installed with macports
// g++ -O2 -Wall -shared -std=c++20 -undefined dynamic_lookup $(python3 -m pybind11 --includes) -I/opt/local/include -L/opt/local/lib -lgsl -lblas -L../build -lgyronimo -I../include -isysroot`xcrun --show-sdk-path` NEAT.cpp -o NEAT.so

// g++ -O2 -Wall -shared -std=c++20 -undefined dynamic_lookup $(python3 -m pybind11 --includes) -I/opt/local/include -L/opt/local/lib -lgsl -lblas -L../build -lgyronimo -I../external/pybind11/include -I../external/gyronimo/ -Wl,-rpath ../build -isysroot`xcrun --show-sdk-path` NEAT.cc -o NEAT.so

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// // // //  USE CUBIC_GSL_PERIODIC
#include <gyronimo/interpolators/cubic_gsl.hh>
#include <gyronimo/interpolators/steffen_gsl.hh>
#include "metric_stellna_qs.hh"
// #include <gyronimo/metrics/metric_stellna.hh>
#include <gyronimo/core/dblock.hh>
#include <vector>
#include "equilibrium_stellna_qs.hh"
// #include <gyronimo/fields/equilibrium_stellna.hh>
#include <gyronimo/dynamics/guiding_centre.hh>
#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
#include <boost/numeric/odeint/stepper/bulirsch_stoer.hpp>
#include <gyronimo/dynamics/odeint_adapter.hh>
#include <gyronimo/core/codata.hh>
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
PYBIND11_MODULE(Neat, m) {
    m.doc() = "Gyronimo Wrapper for the Stellarator Near-Axis Expansion (STELLNA)";
    // m.def("gc_solver",&gc_solver);
    m.def("gc_solver_qs",&gc_solver_qs);
}
