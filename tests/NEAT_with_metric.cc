// g++ -O3 -Wall -shared -std=c++20 -undefined dynamic_lookup $(python3 -m pybind11 --includes) -I/opt/local/include -L/opt/local/lib -lgsl -lblas -L../build -lgyronimo -I../include -isysroot`xcrun --show-sdk-path` NEAT.cpp -o NEAT.so

// #include <gyronimo/parsers/parser_stellna.hh>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gyronimo/interpolators/cubic_gsl.hh>
#include <gyronimo/metrics/metric_stellna.hh>
#include <gyronimo/core/dblock.hh>
#include <vector>
#include <gyronimo/fields/equilibrium_stellna.hh>
#include <gyronimo/dynamics/guiding_centre.hh>
#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
#include <boost/numeric/odeint/stepper/bulirsch_stoer.hpp>
#include <gyronimo/dynamics/odeint_adapter.hh>
#include <gyronimo/core/codata.hh>
namespace py = pybind11;
using namespace gyronimo;

std::vector<std::vector<std::vector<double>>> metric_info(
  int field_periods, double lprime,
  double B0, double etabar,
  const std::vector<double>& B20,
  double B2c, double B2s, double G0,
  double G2, double I2, double iota,
  double iotaN, double beta1s,
  const std::vector<double>& phi_grid,
  const std::vector<double>& torsion,
  const std::vector<double>& curvature,
  const std::vector<double>& X1c,
  const std::vector<double>& X1s,
  const std::vector<double>& Y1c,
  const std::vector<double>& Y1s,
  const std::vector<double>& X20,
  const std::vector<double>& X2c,
  const std::vector<double>& X2s,
  const std::vector<double>& Y20,
  const std::vector<double>& Y2c,
  const std::vector<double>& Y2s,
  const std::vector<double>& Z20,
  const std::vector<double>& Z2c,
  const std::vector<double>& Z2s,
  const std::vector<std::vector<double>>& position
) {
  auto  metric  = std::vector<std::vector<double>>(position.size(), std::vector<double>(7));
  auto dmetric  = std::vector<std::vector<double>>(position.size(), std::vector<double>(18));
  auto jacobian = std::vector<double>(position.size());

  cubic_gsl_factory ifactory;
  metric_stellna g(field_periods, lprime, dblock_adapter(phi_grid), dblock_adapter(torsion), dblock_adapter(curvature),
                   dblock_adapter(X1c), dblock_adapter(X1s), dblock_adapter(Y1c), dblock_adapter(Y1s),
                   dblock_adapter(X20), dblock_adapter(X2c), dblock_adapter(X2s),
                   dblock_adapter(Y20), dblock_adapter(Y2c), dblock_adapter(Y2s),
                   dblock_adapter(Z20), dblock_adapter(Z2c), dblock_adapter(Z2s),
                   B0, etabar, dblock_adapter(B20), B2c,
                   B2s, G0, G2, I2, iota, iotaN, &ifactory);

  for (long unsigned int i=0; i<position.size(); i++) {
    IR3    pos = {position[i][0], position[i][1], position[i][2]};
    SM3   gpos = g(pos);
    dSM3 dgpos = g.del(pos);
    jacobian[i] = std::sqrt(
      gpos[SM3::uu]*gpos[SM3::vv]*gpos[SM3::ww] + 2.0*gpos[SM3::uv]*gpos[SM3::uw]*gpos[SM3::vw] -
      gpos[SM3::uv]*gpos[SM3::uv]*gpos[SM3::ww] - gpos[SM3::uu]*gpos[SM3::vw]*gpos[SM3::vw] -
      gpos[SM3::uw]*gpos[SM3::uw]*gpos[SM3::vv]);
    metric[i]  = {gpos[SM3::uu],gpos[SM3::vv],gpos[SM3::ww],gpos[SM3::uv],gpos[SM3::uw],gpos[SM3::vw],jacobian[i]};
    dmetric[i] = {dgpos[dSM3::uuu],dgpos[dSM3::uuv],dgpos[dSM3::uuw],
                  dgpos[dSM3::vvu],dgpos[dSM3::vvv],dgpos[dSM3::vvw],
                  dgpos[dSM3::wwu],dgpos[dSM3::wwv],dgpos[dSM3::www],
                  dgpos[dSM3::uvu],dgpos[dSM3::uvv],dgpos[dSM3::uvw],
                  dgpos[dSM3::uwu],dgpos[dSM3::uwv],dgpos[dSM3::uww],
                  dgpos[dSM3::vwu],dgpos[dSM3::vwv],dgpos[dSM3::vww]};
  }

  return {position, metric, dmetric};
}

std::vector< std::vector<double>> gc_solver(
  int field_periods, double lprime,
  double B0, double etabar,
  const std::vector<double>& B20,
  double B2c, double B2s, double G0,
  double G2, double I2, double iota,
  double iotaN, double beta1s,
  const std::vector<double>& phi_grid,
  const std::vector<double>& torsion,
  const std::vector<double>& curvature,
  const std::vector<double>& X1c,
  const std::vector<double>& X1s,
  const std::vector<double>& Y1c,
  const std::vector<double>& Y1s,
  const std::vector<double>& X20,
  const std::vector<double>& X2c,
  const std::vector<double>& X2s,
  const std::vector<double>& Y20,
  const std::vector<double>& Y2c,
  const std::vector<double>& Y2s,
  const std::vector<double>& Z20,
  const std::vector<double>& Z2c,
  const std::vector<double>& Z2s,
  double charge, double rhom, double mass,
  double lambda, double energy, double r0,
  double theta0, double phi0, double nsamples,
  double R0, double Tfinal
) {
  // Compute normalisation constants:
  double Valfven = B0/std::sqrt(gyronimo::codata::mu0*(rhom*gyronimo::codata::m_proton*1.e+19));
  double Ualfven = 0.5*gyronimo::codata::m_proton*mass*Valfven*Valfven;
  double energySI = energy*gyronimo::codata::e;
  double Lref = R0;
  double vpp_sign = std::copysign(1.0, lambda);

  // Prepare metric, equilibrium and particles
  cubic_gsl_factory ifactory;
  metric_stellna g(field_periods, lprime, dblock_adapter(phi_grid), dblock_adapter(torsion), dblock_adapter(curvature),
                   dblock_adapter(X1c), dblock_adapter(X1s), dblock_adapter(Y1c), dblock_adapter(Y1s),
                   dblock_adapter(X20), dblock_adapter(X2c), dblock_adapter(X2s),
                   dblock_adapter(Y20), dblock_adapter(Y2c), dblock_adapter(Y2s),
                   dblock_adapter(Z20), dblock_adapter(Z2c), dblock_adapter(Z2s),
                   B0, etabar, dblock_adapter(B20), B2c,
                   B2s, G0, G2, I2, iota, iotaN, &ifactory);
  equilibrium_stellna qsc(&g, B0, etabar, dblock_adapter(B20), dblock_adapter(phi_grid), B2c, B2s, G0, G2, I2, iota, iotaN, beta1s, lprime, &ifactory);
  guiding_centre gc(Lref, Valfven, charge/mass, std::abs(lambda)*energySI/Ualfven, &qsc);
  guiding_centre::state initial_state = gc.generate_state(
      {r0, theta0, phi0}, energySI/Ualfven,(vpp_sign > 0 ? gyronimo::guiding_centre::plus : gyronimo::guiding_centre::minus));
  std::cout << gc.energy_parallel(initial_state);

  // Define variables for integration
  std::vector<std::vector< double >> x_vec;
  class push_back_state_and_time{
  public:
    std::vector< std::vector< double > >& m_states;
    push_back_state_and_time( std::vector< std::vector< double > > &states, 
      const IR3field_c1* e, const guiding_centre* g)
    : m_states( states ), eq_pointer_(e), gc_pointer_(g) { }
    void operator()( const guiding_centre::state& s , double t ){
      IR3 x = gc_pointer_->get_position(s);
      double B = eq_pointer_->magnitude(x, t);
      guiding_centre::state dots = (*gc_pointer_)(s, t);
      IR3 y = gc_pointer_->get_position(dots);
      m_states.push_back({t,x[0],x[1],x[2],
        gc_pointer_->energy_parallel(s),
        gc_pointer_->energy_perpendicular(s, t),
        B, gc_pointer_->get_vpp(s), y[0], y[1], y[2],
        gc_pointer_->get_vpp(dots)});
    }
  private:
    const IR3field_c1* eq_pointer_;
    const guiding_centre* gc_pointer_;
  };

  // Integrate for t in [0,Tfinal], with dt=Tfinal/nsamples, using RK4.
  boost::numeric::odeint::bulirsch_stoer<guiding_centre::state> integration_algorithm;
  //boost::numeric::odeint::runge_kutta4<gyronimo::guiding_centre::state> integration_algorithm;
  boost::numeric::odeint::integrate_const(
      integration_algorithm, odeint_adapter(&gc),
      initial_state, 0.0, Tfinal, Tfinal/nsamples, push_back_state_and_time(x_vec,&qsc,&gc) );

  return x_vec;
}

PYBIND11_MODULE(NEAT, m) {
    m.doc() = "Near-Axis Stellarator geometry";
    m.def("metric_info",&metric_info);
    m.def("gc_solver",&gc_solver);
}
