#include <omp.h>
#include <cmath>
#include <random>
#include <chrono>
#include <metrics_NEAT/metric_stellna_qs.hh>
#include <fields_NEAT/equilibrium_stellna_qs.hh>
#include <gyronimo/core/codata.hh>
#include <gyronimo/dynamics/guiding_centre.hh>
#include <gyronimo/dynamics/odeint_adapter.hh>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>

// can these two be made automatic?
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

// defines the ensemble size and dynamical system:
typedef ensemble<gyronimo::guiding_centre, 3000> ensemble_type;

// ODEInt observer object to print diagnostics at each time step.
class orbit_observer {
public:
  orbit_observer(const gyronimo::guiding_centre* g) : gc_pointer_(g) {};
  void operator()(const ensemble_type::state& z, double t) {
    std::cout  << t << " ";
    for(std::size_t k = 0;k < ensemble_type::size;k++){
      gyronimo::IR3 x = gc_pointer_->get_position(z[k]);
    //   double v_parallel = gc_pointer_->get_vpp(z[k]);
      std::cout
        << x[gyronimo::IR3::u] << " ";
//      << x[gyronimo::IR3::v] << " "
//      << x[gyronimo::IR3::w] << " "
//      << v_parallel << " ";
    }
    std::cout << "\n";
  };
private:
  const gyronimo::guiding_centre* gc_pointer_;
};

int main() {
  const double R0 = 3.0, B0 = 2.7, qaxis=1.0;  // jet-like parameters;
  //auto q = [](double u){return 1.0 + 2.5*u*u;};  // parabolic safety-factor;
  double G0 = B0 * R0, G2 = 0;
  double I2 = B0 / (R0 * qaxis);
  double iota=1/qaxis, iotaN=iota, Bref=B0, etabar=1/R0;
  double B1c = B0*etabar;
  double B20 = 0;
  double B2c = 0;
  double beta1s = 0;

  metric_stellna_qs g(Bref, G0, G2, I2, iota, iotaN,
                     B0, B1c, B20, B2c, beta1s);

  equilibrium_stellna_qs eq(&g);

// defines the guiding-centre equation system:
  const double mass = 2.0, charge = 1.0, mu = 0.0;  // mu=0 deuteron;
  const double vA =  // on-axis Alfven velocity;
      B0/std::sqrt(gyronimo::codata::mu0*gyronimo::codata::m_proton*mass*5.e19);
  gyronimo::guiding_centre gc(R0, vA, charge/mass, mu, &eq);

// gets the number of threads from the openmp environment:
  omp_set_dynamic(0);  // explicitly disable dynamic teams
  size_t nthreads;
#pragma omp parallel
  nthreads = omp_get_num_threads();

// the clock starts ticking here...
  auto begin = std::chrono::high_resolution_clock::now();

// defines the ensemble initial state:
  ensemble_type::state initial;
  std::mt19937 rand_generator;
  std::uniform_real_distribution<> r_distro(0.01, 0.3), vpp_distro(0.25, 1.25);
#pragma omp parallel for
  for(std::size_t k = 0;k < ensemble_type::size;k++)
    initial[k] = gc.generate_state(
        {r_distro(rand_generator), 0.0, 0.0}, vpp_distro(rand_generator),
        gyronimo::guiding_centre::vpp_sign::plus);

// integrates for t in [0,Tfinal], with dt=Tfinal/nsamples, using RK4.
  const double Tfinal = 50.0;
  const std::size_t nsamples = 1000;
  boost::numeric::odeint::runge_kutta4<ensemble_type::state> ode_stepper;
  boost::numeric::odeint::integrate_const(
      ode_stepper, ensemble_type(&gc),
      initial, 0.0, Tfinal, Tfinal/nsamples //, observer
      );

// prints the elapsed time:
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed_mseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  std::cout << nthreads << " " << elapsed_mseconds.count()/1000.0 << std::endl;

  return 0;
}
