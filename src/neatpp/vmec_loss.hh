#ifdef OPENMP
#include <omp.h>
#endif
#include <cmath>
#include <chrono>
#include <random>
#include <argh.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gyronimo/version.hh>
#include <gyronimo/core/codata.hh>
// #include <gyronimo/core/linspace.hh>
#include <gyronimo/parsers/parser_vmec.hh>
#include <gyronimo/fields/equilibrium_vmec.hh>
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
#include <functional>
#include <boost/functional/hash.hpp>

#include <boost/numeric/odeint/integrate/integrate_const.hpp>
#include <boost/numeric/odeint/integrate/integrate_adaptive.hpp>
#include <gsl/gsl_errno.h>

using namespace gyronimo;
using namespace std;
using namespace boost::numeric::odeint;

// Definition linspace function
template<typename T>
vector<double> vmec_linspace(T start_in, T end_in, int num_in)
{

  static_assert(std::is_same<decltype(start_in), decltype(end_in)>::value, "start_in and end_in must have the same type");

  vector<double> linspaced;
  
  double start = static_cast<double>(start_in);
  double end = static_cast<double>(end_in);
  double num = static_cast<double>(num_in);

  if (num < 0) { throw std::invalid_argument("num_in must be non-negative"); }

  if (num == 0) { return linspaced; }
  if (num == 1) 
    {
      linspaced.push_back(start);
      return linspaced;
    }

  double delta = (end - start) / (num - 1);

  for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
  linspaced.push_back(end); // I want to ensure that start and end
                            // are exactly the same as the input
  return linspaced;
}

// Attempt of parallelizing
// template<typename T>
// std::vector<double> vmec_linspace(T start_in, T end_in, int num_in) {

//     static_assert(std::is_same<decltype(start_in), decltype(end_in)>::value, "start_in and end_in must have the same type");

//     std::vector<double> linspaced;
  
//     double start = static_cast<double>(start_in);
//     double end = static_cast<double>(end_in);
//     double num = static_cast<double>(num_in);

//     if (num < 0) { throw std::invalid_argument("num_in must be non-negative"); }

//     if (num == 0) { return linspaced; }
//     if (num == 1) {
//         linspaced.push_back(start);
//         return linspaced;
//     }

//     double delta = (end - start) / (num - 1);

//     #pragma omp parallel
//     {
//         int tid = omp_get_thread_num();
//         int num_threads = omp_get_num_threads();

//         int chunk_size = (num - 1) / num_threads;
//         int start_idx = tid * chunk_size;
//         int end_idx = (tid == num_threads - 1) ? num - 1 : start_idx + chunk_size;

//         for (int i = start_idx; i < end_idx; i++) {
//             linspaced.push_back(start + delta * i);
//         }
//     }

//     linspaced.push_back(end); // Ensure that start and end are exactly the same as the input

//     return linspaced;
// }

// Definition of random distribution
template<typename T>
vector<double> vmec_rand_dist(T start_in, T end_in, int num_in, int dist){

    vector<double> vec(num_in);
    mt19937 gen(dist);
    uniform_real_distribution<> dis(start_in, end_in);

    // #pragma omp parallel for
    for(int i = 0; i < num_in; i++) {
        vec[i] = dis(gen);
    }
    return vec;
}


// Particle ensemble class. The "Gyrons"! Gyrons are individual guiding-centres.
// When you have more than one Gyron, you obtain an ensemble.
template<typename Gyron>
class vmec_ensemble {
public:
    typedef vector<typename Gyron::state> state;
    int vpps=2; // Could eventually be replaced in the appropriate places -> Factor of 2 is due to vpp + and  -
    vmec_ensemble(const vector<Gyron>& gyron_ensemble)
     : gyron_ensemble_(gyron_ensemble) {};
    void operator()(const state& f, state& dfdx, double t) const {
        
        // #pragma omp parallel for
        for(size_t k = 0; k < gyron_ensemble_.size(); k++) {
            for(size_t j = 0; j < vpps; j++) {
                if (f[j + k*vpps][0] < 0.99 && f[j + k*vpps][0] > 0.01){
                    dfdx[j + k*vpps] = gyron_ensemble_[k](f[j + k*vpps], t);
                }
                else {dfdx[j + k*vpps] = 0*f[j + k*vpps];}
            }
        }
    };
    size_t size() const { return gyron_ensemble_.size()*vpps; };
    const vector<Gyron>& gyron_ensemble() const { return gyron_ensemble_; };
private:
    const vector<Gyron> gyron_ensemble_;
};

// Observer class to store the particle position and velocity
// at each time step in vmec_ensemble particle tracing functions
typedef vmec_ensemble<guiding_centre> vmec_ensemble_type;
class vmec_orbit_observer {
public:
    vector< vector< double > >& m_states;
    vmec_orbit_observer( vector< vector< double > > &states, const vmec_ensemble_type& particle_ensemble)
        : m_states( states ), particle_ensemble_(particle_ensemble) { };
    void operator()(const vmec_ensemble_type::state& z, double t) {
        vector<double> temp;
        temp.push_back(t);
        for(size_t k = 0; k < particle_ensemble_.size(); k++) {
            IR3 x = particle_ensemble_.gyron_ensemble()[k/2].get_position(z[k]); // Factor of 2 is due to vpp + and  -
            temp.push_back(x[0]);
        }
        m_states.push_back(temp);
    };
private:
    const vmec_ensemble_type& particle_ensemble_;
};

class cached_metric : public gyronimo::metric_vmec {
 public:
  cached_metric(
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

class cached_field : public gyronimo::equilibrium_vmec {
 public:
  cached_field(
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

vector<vector<double>> vmecloss(
    string vmec_file,  double maximum_s, int integrator,
    double charge, double mass, double energy, 
    size_t nlambda_trapped, size_t nlambda_passing, double r0, double r_max,
    size_t ntheta, size_t nphi, size_t nsamples, double Tfinal, size_t nthreads, size_t dist)
{
    omp_set_dynamic(0);
    omp_set_num_threads(nthreads);
    parser_vmec vmap(vmec_file);
    cubic_gsl_factory ifactory;
    // metric_vmec g(&vmap, &ifactory);
    cached_metric g(&vmap, &ifactory);
    // equilibrium_vmec veq(&g, &ifactory);
    cached_field veq(&g, &ifactory);

    double Lref = 1.0;
    double Vref = 1.0;
    double refEnergy = 0.5 * codata::m_proton * mass * Vref * Vref;
    double energySI = energy * codata::e;
    double energySI_over_refEnergy = energySI / refEnergy;

    // double Bref = vmap.B_0();
    vector<double> lambda_trapped(nlambda_trapped);
    vector<double> lambda_passing(nlambda_passing);
    vector<double> theta(ntheta);
    vector<double> phi(nphi);

    if (dist==0) {
        theta = vmec_linspace(0.0, 2*numbers::pi, ntheta);
        phi = vmec_linspace(0.0, 2*numbers::pi/4, nphi);
        // double nfp=4;
        // phi = vmec_linspace(0.0, 2*M_PI/nfp, nphi);
        double threshold=0.75;
        lambda_trapped = vmec_linspace(threshold, 0.99, nlambda_trapped);
        lambda_passing = vmec_linspace(0.0, threshold, nlambda_passing);
        // lambda_trapped = vmec_linspace(Bref/B_max, Bref/B_min, nlambda_trapped);
        // lambda_passing = vmec_linspace(0.0, Bref/B_max*(1.0-1.0/nlambda_passing), nlambda_passing);
    }
    else {
        theta = vmec_rand_dist(0.0, 2*M_PI, ntheta, dist);
        phi = vmec_rand_dist(0.0, 2*M_PI/4, nphi, dist);
        // double nfp=4;
        // phi = vmec_rand_dist(0.0, 2*M_PI/nfp, nphi, dist);
        double threshold=0.75;
        lambda_trapped = vmec_rand_dist(threshold, 0.99, nlambda_trapped, dist);
        lambda_passing = vmec_rand_dist(0.0, threshold, nlambda_passing, dist);
        // lambda_trapped = vmec_rand_dist(Bref/B_max, Bref/B_min, nlambda_trapped, dist);
        // lambda_passing = vmec_rand_dist(0.0, Bref/B_max*(1.0-1.0/nlambda_passing, dist), nlambda_passing);
    }
    
    vector<double> lambdas = lambda_trapped;
    lambdas.insert(lambdas.end(),lambda_passing.begin(),lambda_passing.end());

    vector<guiding_centre> guiding_centre_vector;
    ensemble_type::state initial;
    //   #pragma omp parallel for
    for(size_t j = 0; j < ntheta; j++) {
        for(size_t l = 0; l < nphi; l++) {
            double Bi=veq.magnitude({r0, phi[l], theta[j]}, 0);
            // cout << Bi << ' ' << r0 << ' ' << theta[j] << ' ' << phi[l] << ' ' << endl;
            for(size_t k = 0; k < nlambda_trapped + nlambda_passing; k++) {
                auto GC=guiding_centre(Lref, Vref, charge/mass, lambdas[k]*energySI_over_refEnergy/Bi, &veq);
                initial.push_back(GC.generate_state(
                {r0, phi[l], theta[j]}, energySI_over_refEnergy,guiding_centre::vpp_sign::plus));
                initial.push_back(GC.generate_state(
                {r0, phi[l], theta[j]}, energySI_over_refEnergy,guiding_centre::vpp_sign::minus));
                guiding_centre_vector.push_back(move(GC));
            }
        }
    }
    cout.precision(16);
    cout.setf(ios::scientific);
    vector<vector<double>> x_vec;
    
    vmec_ensemble_type ensemble_object(move(guiding_centre_vector));
    runge_kutta_cash_karp54<vmec_ensemble_type::state> integration_algorithm;
    // runge_kutta_fehlberg78<vmec_ensemble_type::state> integration_algorithm;
    vmec_orbit_observer observer(x_vec, ensemble_object);
    
    integrate_const(integration_algorithm, ensemble_object,
        initial, 0.0, Tfinal, Tfinal / nsamples, observer);
    // auto start_time = std::chrono::steady_clock::now();
    // integrate_const(integration_algorithm, ensemble_object,
    //     initial, 0.0, Tfinal, Tfinal / nsamples);
    // auto end_time = std::chrono::steady_clock::now();

    // auto duration_s = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // std::cout << "Execution time: " << duration_s << "s" << std::endl;
    
    gsl_set_error_handler_off();
    return x_vec;
}
