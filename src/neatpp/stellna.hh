#include <omp.h>
#include <numbers>
#include <random>
#include <chrono>
using namespace std;
// Define multiplication by scalar and vector sum
array<double,4> operator*(const double& a, const array<double,4>& v) {
    array<double, 4> result = {a*v[0], a*v[1], a*v[2], a*v[3]};
    return result;
}
array<double,4> operator+(
    const array<double,4>& u, const array<double,4>& v) {
    array<double, 4> result = {u[0]+v[0],u[1]+v[1],u[2]+v[2],u[3]+v[3]};
    return result;
}
// #include <boost/numeric/odeint.hpp>
// #include <boost/numeric/odeint/stepper/generation/make_controlled.hpp>
#include <boost/numeric/odeint.hpp> // odeint function definitions
#include <boost/functional/hash.hpp>
// #include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
// #include <boost/numeric/odeint/stepper/adams_bashforth_moulton.hpp>
// #include <boost/numeric/odeint/stepper/adams_moulton.hpp>
// #include <boost/numeric/odeint/stepper/runge_kutta_cash_karp54.hpp>
// #include <boost/numeric/odeint/stepper/runge_kutta_fehlberg78.hpp>
// #include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>

// #include <boost/numeric/odeint/algebra/range_algebra.hpp>
// #include <boost/numeric/odeint/algebra/default_operations.hpp>
// #include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>

// #include <boost/numeric/odeint/integrate/integrate_const.hpp>
// #include <boost/numeric/odeint/integrate/integrate_adaptive.hpp>
#include <gyronimo/core/codata.hh>
#include <gyronimo/core/dblock.hh>
#include <gyronimo/interpolators/cubic_gsl.hh>
#include <gyronimo/dynamics/guiding_centre.hh>
#include <gyronimo/dynamics/odeint_adapter.hh>
#include <gyronimo/interpolators/cubic_periodic_gsl.hh>
#include "metric_stellna_qs.hh"
#include "equilibrium_stellna_qs.hh"
#include "metric_stellna_qs_partial.hh"
#include "equilibrium_stellna_qs_partial.hh"
#include "metric_stellna.hh"
#include "equilibrium_stellna.hh"
using namespace gyronimo;
using namespace boost::numeric::odeint;

// Normalization to SI units
double Lref = 1.0;
double Vref = 1.0;
double energySIoverRefEnergy(double mass, double energy){
    double refEnergy = 0.5*codata::m_proton*mass*Vref*Vref;
    double energySI = energy*codata::e;
    return energySI/refEnergy;
}


// Definition linspace function
template<typename T>
vector<double> neat_linspace(T start_in, T end_in, int num_in)
{
  vector<double> linspaced;
  double start = static_cast<double>(start_in);
  double end = static_cast<double>(end_in);
  double num = static_cast<double>(num_in);

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

// Definition of random distribution
template<typename T>
vector<double> neat_rand_dist(T start_in, T end_in, int num_in, int dist){

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
class ensemble {
public:
    typedef vector<typename Gyron::state> state;
    int vpps=2; // This could eventually be replaced in the appropriate places -> Just to account for + and - vpp
    ensemble(const vector<Gyron>& gyron_ensemble)
     : gyron_ensemble_(gyron_ensemble) {};
    void operator()(const state& f, state& dfdx, double t) const {
        #pragma omp parallel for
        for(size_t k = 0; k < gyron_ensemble_.size(); k++) {
            for(size_t j = 0; j < vpps; j++) {
                // This is hardcoding the freezing places -> To change
                if (f[j + k*vpps][0] < 1.7044*sqrt(0.99) && f[j + k*vpps][0] > 0.01){
                    dfdx[j + k*vpps] = gyron_ensemble_[k](f[j + k*vpps], t);}
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
// at each time step in single particle tracing functions
class push_back_state_and_time {
public:
    vector< vector< double > >& m_states;
    push_back_state_and_time( vector< vector< double > > &states,
                                const IR3field_c1* e, const guiding_centre* g)
        : m_states( states ), eq_pointer_(e), gc_pointer_(g) { }
    void operator()(const guiding_centre::state& s, double t) {
        IR3 x = gc_pointer_->get_position(s);
        double B = (eq_pointer_->magnitude(x, t)) * eq_pointer_->m_factor();
        guiding_centre::state dots = (*gc_pointer_)(s, t);
        IR3 y = gc_pointer_->get_position(dots);
        IR3 X = eq_pointer_->metric()->transform2cylindrical(x);
        double v_parallel = gc_pointer_->get_vpp(s);
        IR3 B_cov = (eq_pointer_->covariant(x, t)) * eq_pointer_->m_factor();
        IR3 B_con = (eq_pointer_->contravariant(x, t)) * eq_pointer_->m_factor();

    // std::cout << "STELLNA" << std::endl;

    // dIR3 dB_cov = (eq_pointer_->del_covariant(x, t));
    // dIR3 dB_con = (eq_pointer_->del_contravariant(x, t));
    // double jac = eq_pointer_->metric()->jacobian(x);
    // IR3 del_jac = eq_pointer_->metric()->del_jacobian(x);
    // std::cout << "jac: " << jac << std::endl;
    // std::cout << "G/B^2: " << B_cov[IR3::w]/B/B << std::endl;
    // std::cout << "del_jac_u:" << del_jac[IR3::u] << ", del_jac_v:" << del_jac[IR3::v] << ", del_jac_w:" << del_jac[IR3::w]<< std::endl;
    // std::cout << "B: " << B << std::endl;
    // std::cout << "B_cov[IR3::u]:" << B_cov[IR3::u] << std::endl;
    // std::cout << "B_cov[IR3::v]:" << B_cov[IR3::v] << std::endl;
    // std::cout << "B_cov[IR3::w]:" << B_cov[IR3::w] << std::endl;
    // std::cout << "dB_covw_u:" << dB_cov[dIR3::wu] * eq_pointer_->m_factor() << ", dB_covw_v:" << dB_cov[dIR3::wv] * eq_pointer_->m_factor() << ", dB_covw_w:" << dB_cov[dIR3::ww] * eq_pointer_->m_factor() << std::endl;
    // std::cout << "B_con[IR3::u]:" << B_con[IR3::u] << std::endl;
    // std::cout << "B_con[IR3::v]:" << B_con[IR3::v] << std::endl;
    // std::cout << "B_con[IR3::w]:" << B_con[IR3::w] << std::endl;
    // std::cout << "dB_conv_u:" << dB_con[dIR3::vu] * eq_pointer_->m_factor() << ", dB_conv_v:" << dB_con[dIR3::vv] * eq_pointer_->m_factor() << ", dB_conv_w:" << dB_con[dIR3::vw] * eq_pointer_->m_factor() << std::endl;
    // std::cout << "dB_conw_u:" << dB_con[dIR3::wu] * eq_pointer_->m_factor() << ", dB_conw_v:" << dB_con[dIR3::wv] * eq_pointer_->m_factor() << ", dB_conw_w:" << dB_con[dIR3::ww] * eq_pointer_->m_factor() << std::endl;
    // exit(0);
        m_states.push_back({
            t,x[0],x[1],x[2],
            gc_pointer_->energy_parallel(s),
            gc_pointer_->energy_perpendicular(s, t),
            B, gc_pointer_->get_vpp(s), y[0], y[1], y[2],
            gc_pointer_->get_vpp(dots),
            B_cov[IR3::u], B_cov[IR3::w], B_cov[IR3::v],
            B_con[IR3::u], B_con[IR3::w], B_con[IR3::v]
        });
    }
private:
    const IR3field_c1* eq_pointer_;
    const guiding_centre* gc_pointer_;
};

// Observer class to store the particle positions and velocities
// at each time step in ensemble particle tracing functions
typedef ensemble<guiding_centre> ensemble_type;
class orbit_observer {
public:
    vector< vector< double > >& m_states;
    orbit_observer( vector< vector< double > > &states, const ensemble_type& particle_ensemble)
        : m_states( states ), particle_ensemble_(particle_ensemble) { };
    void operator()(const ensemble_type::state& z, double t) {
        vector<double> temp;
        temp.push_back(t);
        for(size_t k = 0; k < particle_ensemble_.size(); k++) {
            IR3 x = particle_ensemble_.gyron_ensemble()[k/2].get_position(z[k]);
            temp.push_back(x[0]);
        }
        m_states.push_back(temp);
    };
private:
    const ensemble_type& particle_ensemble_;
};

class cached_metric_na : public metric_stellna_qs_partial {
 public:
  cached_metric_na(
      int field_periods, double Bref,
      const dblock& phi_grid, double G0, double G2,
      double I2, double iota, double iotaN,
      double B0,  double B1c,
      const dblock& B20, double B2c, double beta1s,
      const gyronimo::interpolator1d_factory* ifactory)
      : metric_stellna_qs_partial(field_periods, Bref, phi_grid,
        G0, G2, I2, iota, iotaN, B0, B1c, B20, B2c, beta1s, ifactory) {};
  virtual gyronimo::SM3 operator()(const gyronimo::IR3& x) const override {
    thread_local gyronimo::IR3 cx = {0,0,0};
    thread_local gyronimo::SM3 cg = {0,0,0,0,0,0};
    if(boost::hash_value(x) != boost::hash_value(cx)) {
      cg = metric_stellna_qs_partial::operator()(x);
      cx = x;
    }
    return cg;
  };
  virtual gyronimo::dSM3 del(const gyronimo::IR3& x) const override {
    thread_local gyronimo::IR3 cx = {0,0,0};
    thread_local gyronimo::dSM3 cdg = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    if(boost::hash_value(x) != boost::hash_value(cx)) {
      cdg = metric_stellna_qs_partial::del(x);
      cx = x;
    }
    return cdg;
  };
};

class cached_field_na : public equilibrium_stellna_qs_partial {
 public:
  cached_field_na(
      const metric_stellna_qs_partial* m)
      : equilibrium_stellna_qs_partial(m) {};
    virtual gyronimo::IR3 contravariant(
      const gyronimo::IR3& x, double t) const override {
    thread_local double ct = -1;
    thread_local gyronimo::IR3 cx = {0,0,0};
    thread_local gyronimo::IR3 cg = {0,0,0};
    if(boost::hash_value(x) != boost::hash_value(cx) || t != ct) {
      cg = equilibrium_stellna_qs_partial::contravariant(x, t);
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
      cdg = equilibrium_stellna_qs_partial::del_contravariant(x, t);
      cx = x;
      ct = t;
    }
    return cdg;
  };
};


/*****************************************

    SINGLE particle tracing functions

******************************************/

// Fully quasisymmetric
vector< vector<double>> gc_solver_qs(
                                   double G0, double G2, double I2, double nfp, double iota,
                                   double iotaN,
                                   const vector<double>& phi_grid,
                                   double B0, double B1c,
                                   double B20, double B2c, double beta1s, double charge,
                                   double mass, double lambda, int vpp_sign,
                                   double energy, double r0, double theta0,
                                   double phi0, size_t nsamples, double Tfinal
                               )
{

    double Bref = B0;
    double energySI_over_refEnergy = energySIoverRefEnergy(mass, energy);
    metric_stellna_qs g(Bref, G0, G2, I2, iota, iotaN,
                        B0, B1c, B20, B2c, beta1s);
    equilibrium_stellna_qs qsc(&g);
    double Bi = qsc.magnitude({r0, theta0, phi0}, 0);
    guiding_centre gc(Lref, Vref, charge/mass, lambda*energySI_over_refEnergy/Bi, &qsc);  // -> Version with Bi
    // guiding_centre gc(Lref, Vref, charge/mass, lambda*energySI_over_refEnergy/B0, &qsc); // -> Version with B0
    guiding_centre::state initial_state = gc.generate_state(
        {r0, theta0, phi0}, energySI_over_refEnergy,
        (vpp_sign > 0 ? guiding_centre::plus : guiding_centre::minus));

    vector<vector< double >> x_vec;
    runge_kutta4<guiding_centre::state> integration_algorithm;
    runge_kutta_cash_karp54<guiding_centre::state> integration_algorithm2;

    integrate_const(
        integration_algorithm2, odeint_adapter(&gc),
        initial_state, 0.0, Tfinal, Tfinal/nsamples, push_back_state_and_time(x_vec,&qsc,&gc) );

    return x_vec;
}

// Partially quasisymmetric
vector< vector<double>> gc_solver_qs_partial(
                                   double G0, double G2, double I2, double nfp,
                                   double iota, double iotaN,
                                   const vector<double>& phi_grid,
                                   double B0, double B1c,
                                   const vector<double>& B20,
                                   double B2c, double beta1s, double charge,
                                   double mass, double lambda, int vpp_sign,
                                   double energy, double r0, double theta0,
                                   double phi0, size_t nsamples, double Tfinal
                               )
{
    double Bref = B0;
    double energySI_over_refEnergy = energySIoverRefEnergy(mass, energy);
    cubic_periodic_gsl_factory ifactory;

    metric_stellna_qs_partial g(nfp, Bref, dblock_adapter(phi_grid), G0, G2, I2, iota, iotaN, B0, B1c,
                                dblock_adapter(B20), B2c, beta1s, &ifactory);
    equilibrium_stellna_qs_partial qsc(&g);
    double Bi = qsc.magnitude({r0, theta0, phi0}, 0);
    guiding_centre gc(Lref, Vref, charge/mass, lambda*energySI_over_refEnergy/Bi, &qsc); 
    guiding_centre::state initial_state = gc.generate_state(
    {r0, theta0, phi0}, energySI_over_refEnergy,(vpp_sign > 0 ? guiding_centre::plus : guiding_centre::minus));

    vector<vector< double >> x_vec;
    // runge_kutta4<guiding_centre::state> integration_algorithm;
    runge_kutta_dopri5<guiding_centre::state> integration_algorithm2;
    // typedef guiding_centre::state state_type;
    // typedef runge_kutta_cash_karp54<state_type> error_stepper_type;
    // double abs_err = 1.0e-10 , rel_err = 1.0e-6 , a_x = 1.0 , a_dxdt = 1.0;
    integrate_const(integration_algorithm2, odeint_adapter(&gc),
        initial_state, 0.0, Tfinal, Tfinal/nsamples, push_back_state_and_time(x_vec,&qsc,&gc) );
    // integrate_adaptive(
    //     make_controlled( 1.0e-14 , 1.0e-16 , error_stepper_type() ), odeint_adapter(&gc),
    //     initial_state, 0.0, Tfinal, Tfinal/nsamples, push_back_state_and_time(x_vec,&qsc,&gc) );

    return x_vec;
}

// General field
vector< vector<double>> gc_solver(
                                   int field_periods,
                                   double G0, double G2, double I2,
                                   double iota, double iotaN, double Bref,
                                   const vector<double>& phi_grid,
                                   const vector<double>& B0,
                                   const vector<double>& B1c,
                                   const vector<double>& B1s,
                                   const vector<double>& B20,
                                   const vector<double>& B2c,
                                   const vector<double>& B2s,
                                   const vector<double>& beta0,
                                   const vector<double>& beta1c,
                                   const vector<double>& beta1s,
                                   double charge, double mass, double lambda,
                                   int vpp_sign, double energy, double r0, double theta0,
                                   double phi0, size_t nsamples, double Tfinal)
{
    double energySI_over_refEnergy = energySIoverRefEnergy(mass, energy);
    cubic_periodic_gsl_factory ifactory;

    metric_stellna g(field_periods, Bref, dblock_adapter(phi_grid), G0, G2, I2, iota, iotaN,
                     dblock_adapter(B0), dblock_adapter(B1c), dblock_adapter(B1s),
                     dblock_adapter(B20), dblock_adapter(B2c), dblock_adapter(B2s),
                     dblock_adapter(beta0), dblock_adapter(beta1c), dblock_adapter(beta1s),
                     &ifactory);
    equilibrium_stellna qsc(&g);

    double Bi = qsc.magnitude({r0, theta0, phi0}, 0);
    
    guiding_centre gc(Lref, Vref, charge/mass, lambda*energySI_over_refEnergy/Bi, &qsc);
    guiding_centre::state initial_state = gc.generate_state(
        {r0, theta0, phi0}, energySI_over_refEnergy,
        (vpp_sign > 0 ? guiding_centre::plus : guiding_centre::minus));

    vector<vector< double >> x_vec;
    // runge_kutta4<guiding_centre::state> integration_algorithm;
    runge_kutta_fehlberg78<guiding_centre::state> integration_algorithm2;

    integrate_const(integration_algorithm2, odeint_adapter(&gc),
        initial_state, 0.0, Tfinal, Tfinal/nsamples, push_back_state_and_time(x_vec,&qsc,&gc) );

    return x_vec;
}

/****************************************

    ENSEMBLE particle tracing functions

*****************************************/

// Fully quasisymmetric ensemble
tuple<vector<double>,vector<vector<double>>> gc_solver_qs_ensemble(
                                   double G0, double G2, double I2, double nfp, double iota,
                                   double iotaN, const vector<double>& phi_grid,
                                   double B0, double B1c, double B20, double B2c,
                                   double beta1s, double charge, double mass, double energy,
                                   size_t nlambda_trapped, size_t nlambda_passing, double r0, double r_max,
                                   size_t ntheta, size_t nphi, size_t nsamples, double Tfinal, size_t nthreads, size_t dist, 
                                   vector<double>& theta, vector<double>& phi
                               )
{

    omp_set_dynamic(0);
    omp_set_num_threads(nthreads);

    double Bref = B0;
    double energySI_over_refEnergy = energySIoverRefEnergy(mass, energy);
    metric_stellna_qs g(Bref, G0, G2, I2, iota, iotaN,
                        B0, B1c, B20, B2c, beta1s);

    equilibrium_stellna_qs qsc(&g);

    double B_max = abs(B0) + abs(r_max * B1c) + r_max * r_max * (abs(B20) + abs(B2c));
    double B_min = max( 0.01, abs(B0) - abs(r_max * B1c) - r_max * r_max * (abs(B20) + abs(B2c)) );

    vector<double> lambda_trapped(nlambda_trapped);
    vector<double> lambda_passing(nlambda_passing);

    double threshold=0.75;

    if (dist==0) {

        lambda_trapped = neat_linspace(threshold, 0.99, nlambda_trapped);
        lambda_passing = neat_linspace(0.0, threshold, nlambda_passing);
        // lambda_trapped = neat_linspace(Bref/B_max, Bref/B_min, nlambda_trapped);
        // lambda_passing = neat_linspace(0.0, Bref/B_max*(1.0-1.0/nlambda_passing), nlambda_passing);
    }
    else {
        lambda_trapped = neat_rand_dist(threshold, 0.99, nlambda_trapped, dist);
        lambda_passing = neat_rand_dist(0.0, threshold, nlambda_passing, dist);
        // lambda_trapped = neat_rand_dist(Bref/B_max, Bref/B_min, nlambda_trapped, dist);
        // lambda_passing = neat_rand_dist(0.0, Bref/B_max*(1.0-1.0/nlambda_passing), nlambda_passing, dist);
    }


    vector<double> lambdas = lambda_trapped;
    lambdas.insert(lambdas.end(),lambda_passing.begin(),lambda_passing.end());

    vector<guiding_centre> guiding_centre_vector;
    ensemble_type::state initial;
    // 
    for(size_t j = 0; j < ntheta; j++) {
        for(size_t l = 0; l < nphi; l++) {
            double Bi=qsc.magnitude({r0, theta[j], phi[l]}, 0);
            // cout << Bi << ' ' << r0 << ' ' << theta[j] << ' ' << phi[l] << ' ' << endl;
            for(size_t k = 0; k < nlambda_trapped + nlambda_passing; k++) {
                auto GC=guiding_centre(Lref, Vref, charge/mass, lambdas[k]*energySI_over_refEnergy/Bi, &qsc);
                initial.push_back(GC.generate_state(
                {r0, theta[j], phi[l]}, energySI_over_refEnergy,guiding_centre::vpp_sign::plus));
                initial.push_back(GC.generate_state(
                {r0, theta[j], phi[l]}, energySI_over_refEnergy,guiding_centre::vpp_sign::minus));
                guiding_centre_vector.push_back(move(GC));
            }
        }
    }
    vector<vector< double >> x_vec;
    ensemble_type ensemble_object(move(guiding_centre_vector));
    boost::numeric::odeint::runge_kutta4<ensemble_type::state> ode_stepper;
    boost::numeric::odeint::integrate_const(
        ode_stepper, ensemble_object,
        initial, 0.0, Tfinal, Tfinal/nsamples, orbit_observer(x_vec, ensemble_object)
    );
    return make_tuple(lambdas,x_vec);
}

// Partially quasisymmetric ensemble
tuple<vector<double>,vector<vector<double>>> gc_solver_qs_partial_ensemble(
                                   double G0, double G2, double I2, double nfp, double iota,
                                   double iotaN, const vector<double>& phi_grid,
                                   double B0, double B1c, vector<double>& B20, double B2c,
                                   double beta1s, double charge, double mass, double energy,
                                   size_t nlambda_trapped, size_t nlambda_passing, double r0, double r_max,
                                   size_t ntheta, size_t nphi, size_t nsamples, double Tfinal, size_t nthreads, size_t dist, 
                                   vector<vector<double>>& theta, vector<vector<double>>& phi
                               )
{

    omp_set_dynamic(0);
    omp_set_num_threads(nthreads);

    double Bref = B0;
    double energySI_over_refEnergy = energySIoverRefEnergy(mass, energy);
    cubic_periodic_gsl_factory ifactory;

    // metric_stellna_qs_partial g(nfp, Bref, dblock_adapter(phi_grid), G0, G2, I2, iota, iotaN, B0, B1c,
    //                             dblock_adapter(B20), B2c, beta1s, &ifactory);
    // equilibrium_stellna_qs_partial qsc(&g);
    cached_metric_na g(nfp, Bref, dblock_adapter(phi_grid), G0, G2, I2, iota, iotaN, B0, B1c,
                                dblock_adapter(B20), B2c, beta1s, &ifactory);
    cached_field_na qsc(&g);

    double B20_max = abs(*max_element(B20.begin(), B20.end()));
    double B_max = abs(B0) + abs(r_max * B1c) + r_max * r_max * (B20_max + abs(B2c));
    double B_min = max( 0.01, abs(B0) - abs(r_max * B1c) - r_max * r_max * (B20_max + abs(B2c)) );

    vector<double> lambda_trapped(nlambda_trapped);
    vector<double> lambda_passing(nlambda_passing);
    double threshold=0.75;

    if (dist==0) {
        lambda_trapped = neat_linspace(threshold, 0.99, nlambda_trapped);
        lambda_passing = neat_linspace(0.0, threshold, nlambda_passing);
        // lambda_trapped = neat_linspace(Bref/B_max, Bref/B_min, nlambda_trapped);
        // lambda_passing = neat_linspace(0.0, Bref/B_max*(1.0-1.0/nlambda_passing), nlambda_passing);
    }
    else {
        lambda_trapped = neat_rand_dist(threshold, 0.99, nlambda_trapped, dist);
        lambda_passing = neat_rand_dist(0.0, threshold, nlambda_passing, dist);
        // lambda_trapped = neat_rand_dist(Bref/B_max, Bref/B_min, nlambda_trapped, dist);
        // lambda_passing = neat_rand_dist(0.0, Bref/B_max*(1.0-1.0/nlambda_passing), nlambda_passing, dist);
    }

    vector<double> lambdas = lambda_trapped;
    lambdas.insert(lambdas.end(),lambda_passing.begin(),lambda_passing.end());

    vector<guiding_centre> guiding_centre_vector;
    ensemble_type::state initial;
// 
    for(size_t j = 0; j < ntheta; j++) {
        for(size_t l = 0; l < nphi; l++) {
            double Bi=qsc.magnitude({r0, theta[j][l], phi[j][l]}, 0);
            // cout << Bi << ' ' << r0 << ' ' << theta[j] << ' ' << phi[j][l] << ' ' << endl;
            for(size_t k = 0; k < nlambda_trapped + nlambda_passing; k++) {
                auto GC=guiding_centre(Lref, Vref, charge/mass, lambdas[k]*energySI_over_refEnergy/Bi, &qsc);
                initial.push_back(GC.generate_state(
                {r0, theta[j][l], phi[j][l]}, energySI_over_refEnergy,guiding_centre::vpp_sign::plus));
                initial.push_back(GC.generate_state(
                {r0, theta[j][l], phi[j][l]}, energySI_over_refEnergy,guiding_centre::vpp_sign::minus));
                guiding_centre_vector.push_back(move(GC));
            }
        }
    }
    vector<vector< double >> x_vec;
    ensemble_type ensemble_object(move(guiding_centre_vector));
    runge_kutta_cash_karp54<ensemble_type::state> ode_stepper;
    
    boost::numeric::odeint::integrate_const( ode_stepper, ensemble_object,
        initial, 0.0, Tfinal, Tfinal/nsamples, orbit_observer(x_vec, ensemble_object));

    return make_tuple(lambdas,x_vec);
}

// General field ensemble
tuple<vector<double>,vector<vector<double>>> gc_solver_ensemble(
                                   int nfp, double G0, double G2, double I2, double iota,
                                   double iotaN, double Bref, const vector<double>& phi_grid,
                                   const vector<double>& B0,
                                   const vector<double>& B1c,
                                   const vector<double>& B1s,
                                   const vector<double>& B20,
                                   const vector<double>& B2c,
                                   const vector<double>& B2s,
                                   const vector<double>& beta0,
                                   const vector<double>& beta1c,
                                   const vector<double>& beta1s,
                                   double charge, double mass, double energy,
                                   size_t nlambda_trapped, size_t nlambda_passing, double r0, double r_max,
                                   size_t ntheta, size_t nphi, size_t nsamples, double Tfinal, size_t nthreads, size_t dist, 
                                   vector<double>& theta, vector<double>& phi
                               )
{

    omp_set_dynamic(0); 
    omp_set_num_threads(nthreads);

    cubic_periodic_gsl_factory ifactory;

    metric_stellna g(nfp, Bref, dblock_adapter(phi_grid), G0, G2, I2, iota, iotaN,
                     dblock_adapter(B0), dblock_adapter(B1c), dblock_adapter(B1s),
                     dblock_adapter(B20), dblock_adapter(B2c), dblock_adapter(B2s),
                     dblock_adapter(beta0), dblock_adapter(beta1c), dblock_adapter(beta1s),
                     &ifactory);

    equilibrium_stellna qsc(&g);

    double energySI_over_refEnergy = energySIoverRefEnergy(mass, energy);
    double B0_max = abs(*max_element(B0.begin(), B0.end()));
    double B1c_max = abs(*max_element(B1c.begin(), B1c.end()));
    double B1s_max = abs(*max_element(B1s.begin(), B1s.end()));
    double B20_max = abs(*max_element(B20.begin(), B20.end()));
    double B2c_max = abs(*max_element(B2c.begin(), B2c.end()));
    double B2s_max = abs(*max_element(B2s.begin(), B2s.end()));
    double B_max = B0_max + r_max * (B1c_max + B1s_max) + r_max * r_max * (B20_max + B2c_max + B2s_max);
    double B_min = max( 0.01, B0_max - r_max * (B1c_max + B1s_max) - r_max * r_max * (B20_max + B2c_max + B2s_max));

    vector<double> lambda_trapped(nlambda_trapped);
    vector<double> lambda_passing(nlambda_passing);
    double threshold=0.7;

    if (dist==0) {
        lambda_trapped = neat_linspace(threshold, 1.0, nlambda_trapped);
        lambda_passing = neat_linspace(0.0, threshold, nlambda_passing);
        // lambda_trapped = neat_linspace(Bref/B_max, Bref/B_min, nlambda_trapped);
        // lambda_passing = neat_linspace(0.0, Bref/B_max*(1.0-1.0/nlambda_passing), nlambda_passing);
    }
    else {
        lambda_trapped = neat_rand_dist(threshold, 1.0, nlambda_trapped, dist);
        lambda_passing = neat_rand_dist(0.0, threshold, nlambda_passing, dist);
        // lambda_trapped = neat_rand_dist(Bref/B_max, Bref/B_min, nlambda_trapped, dist);
        // lambda_passing = neat_rand_dist(0.0, Bref/B_max*(1.0-1.0/nlambda_passing), nlambda_passing, dist);
    }

    
    vector<double> lambdas = lambda_trapped;
    lambdas.insert(lambdas.end(),lambda_passing.begin(),lambda_passing.end());

    vector<guiding_centre> guiding_centre_vector;
    ensemble_type::state initial;
// 
    for(size_t j = 0; j < ntheta; j++) {
        for(size_t l = 0; l < nphi; l++) {
            double Bi=qsc.magnitude({r0, theta[j], phi[l]}, 0);
            for(size_t k = 0; k < nlambda_trapped + nlambda_passing; k++) {
                auto GC=guiding_centre(Lref, Vref, charge/mass, lambdas[k]*energySI_over_refEnergy/Bi, &qsc);
                initial.push_back(GC.generate_state(
                {r0, theta[j], phi[l]}, energySI_over_refEnergy,guiding_centre::vpp_sign::plus));
                initial.push_back(GC.generate_state(
                {r0, theta[j], phi[l]}, energySI_over_refEnergy,guiding_centre::vpp_sign::minus));
                guiding_centre_vector.push_back(move(GC));
            }
        }
    }
    vector<vector< double >> x_vec;
    ensemble_type ensemble_object(move(guiding_centre_vector));
    boost::numeric::odeint::runge_kutta4<ensemble_type::state> ode_stepper;
    boost::numeric::odeint::integrate_const( ode_stepper, ensemble_object,
        initial, 0.0, Tfinal, Tfinal/nsamples, orbit_observer(x_vec, ensemble_object));

    return make_tuple(lambdas,x_vec);
}
