#include <cmath>
#include <random>
#include <chrono>
#include <execution>
#include <gyronimo/core/codata.hh>

#include <gyronimo/metrics/metric_polar_torus.hh>
#include <gyronimo/fields/equilibrium_circular.hh>

//#include <gyronimo/interpolators/cubic_gsl.hh>
//#include <gyronimo/fields/equilibrium_vmec.hh>

#include <gyronimo/dynamics/guiding_centre.hh>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>

/*
#include <boost/flyweight.hpp>

//! encapsulates a Gyron equation system within a boost::flyweight
template<typename Gyron>
class gyron_flyweight {
 public:
  typedef Gyron::state state;
  gyron_flyweight() = default;
  gyron_flyweight(const Gyron& g) : flyweight_object_(g) {};
  state operator()(const state& f, double t) const {
    return flyweight_object_.get()(f, t);};
  template<typename... Args>
  state generate_state(Args&&... args) const {
    return flyweight_object_.get().generate_state(args...);};
  auto get() const {return flyweight_object_.get();};
 private:
  boost::flyweight<Gyron> flyweight_object_;
};

namespace gyronimo {

size_t hash_value(const guiding_centre& gc) {
  size_t seed = 0;
  boost::hash_combine(seed, gc.mu_tilde());
  boost::hash_combine(seed, gc.qom_tilde());
  return seed;
}

bool operator==(const guiding_centre& gc1, const guiding_centre& gc2) {
  return hash_value(gc1) == hash_value(gc2);
}

}
*/
auto random_distribution(
    size_t size, double Lref, double Vref, const gyronimo::IR3field_c1* eqp) {
  double qom = 0.5;  // for deuterons or alphas;
  std::mt19937 rand_generator;
  std::uniform_real_distribution<>
      r_distro(0.1, 0.9), vpp_distro(0.25, 1.25), mu_distro(0.1, 0.5);
  auto parameter_generator = [&]() {
    return std::tuple(r_distro(rand_generator),
        mu_distro(rand_generator), vpp_distro(rand_generator));};

//typedef gyron_flyweight<gyronimo::guiding_centre> gyron_t;
  typedef gyronimo::guiding_centre gyron_t;
  std::vector<gyron_t::state> initial(size);
  std::vector<gyron_t> gc_set(size);
  std::for_each(std::execution::seq, gc_set.begin(), gc_set.end(),
      [&](gyron_t& gc) {
        auto [r, mu, vpp] = parameter_generator();
        gc = gyronimo::guiding_centre(Lref, Vref, qom, mu, eqp);
        gyronimo::IR3 position = {r, 0, 0};
        initial[&gc - gc_set.data()] = gc.generate_state(
            position, vpp, gyronimo::guiding_centre::vpp_sign::plus);
      });
  return std::pair(gc_set, initial);
}

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

//! Builds a single Gyron object from a collection of Gyrons
template<typename GyronRange>
class ensemble {
 public:
  typedef typename std::ranges::range_value_t<GyronRange> gyron_t;
  typedef typename gyron_t::state gyron_state_t;
  ensemble(const GyronRange& gyron_ensemble)
    : gyron_ensemble_(gyron_ensemble) {};
  template<typename StateRange>
  void operator()(
      const StateRange& f, StateRange& dfdt, double t) const {
    std::transform(std::execution::seq,
        f.begin(), f.end(), gyron_ensemble_.begin(), dfdt.begin(),
            [&t](const gyron_state_t& s, const gyron_t& g) {return g(s, t);});
  };
  size_t size() const {return std::size(gyron_ensemble_);};
  auto begin() const {return std::begin(gyron_ensemble_);};
  auto end() const {return std::end(gyron_ensemble_);};
 private:
  const GyronRange& gyron_ensemble_;
};

// ODEInt observer object to print diagnostics at each time step.
template<typename EnsembleDyn>
class orbit_observer {
 public:
  typedef EnsembleDyn::gyron_t gyron_t;
  typedef EnsembleDyn::gyron_state_t gyron_state_t;
  orbit_observer(const EnsembleDyn& ed) : ensemble_dyn_(ed) {};
  template<typename ensemble_state_t>
  void operator()(const ensemble_state_t& state_ensemble, double t) {
    std::cout  << t << " ";
    std::transform(
        state_ensemble.begin(), state_ensemble.end(), ensemble_dyn_.begin(),
            std::ostream_iterator<double>(std::cout, " "),
                [&t](const gyron_state_t& s, const gyron_t& g) {
//                auto gg = g.get();
                  auto gg = g;
//                return gg.energy_parallel(s) + gg.energy_perpendicular(s, t);
//                auto x = g.get().get_position(s);
                  auto x = g.get_position(s);
                  return x[gyronimo::IR3::u];
                });
    std::cout << "\n";
  };
 private:
  const EnsembleDyn& ensemble_dyn_;
};

int main() {
  const double a = 1.0, R0 = 3.0, B0 = 2.7;  // jet-like parameters;
  auto q = [](double u){return 1.0 + 2.5*u*u;};  // parabolic safety-factor;
  auto qprime = [](double u){return 5.0*u;};  // safety-factor derivative;
  gyronimo::metric_polar_torus g(a, R0);
  gyronimo::equilibrium_circular eq(B0, &g, q, qprime);
/*
  gyronimo::cubic_gsl_factory ifactory;
  gyronimo::parser_vmec p("/Users/prod/Desktop/w7x_standard.nc");
  gyronimo::metric_vmec g(&p, &ifactory);
  gyronimo::equilibrium_vmec eq(&g, &ifactory);
  double B0 = p.B_0(), R0 = p.R_0();
*/
  const double vA =  // on-axis Alfven velocity for normalisation purposes;
      B0/std::sqrt(gyronimo::codata::mu0*gyronimo::codata::m_proton*2.0*5.0e19);

  auto begin = std::chrono::high_resolution_clock::now();  // starts ticking...

  auto [equation_set, initial_state] = random_distribution(1000, R0, vA, &eq);
  ensemble dynamical_system(equation_set);
  orbit_observer observer(dynamical_system);

// integrates for t in [0,Tfinal], with dt=Tfinal/nsamples, using RK4.
  const double Tfinal = 0.01;
  const std::size_t nsamples = 1030;
  boost::numeric::odeint::runge_kutta4<decltype(initial_state)> ode_stepper;
  boost::numeric::odeint::integrate_const(
      ode_stepper, dynamical_system,
      initial_state, 0.0, Tfinal, Tfinal/nsamples //, observer
      );

  auto end = std::chrono::high_resolution_clock::now();  // stops ticking...
  auto elapsed_mseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
  std::cerr << elapsed_mseconds.count()/1000.0 << " sec.\n";

  return 0;
}
