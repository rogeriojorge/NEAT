#ifndef NEATPP
#define NEATPP

#include <omp.h>
#include <numbers>
#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>
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
using namespace std;

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
std::vector<double> neat_linspace(T start_in, T end_in, int num_in)
{

  std::vector<double> linspaced;

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

// Particle ensemble class. The "Gyrons"!
template<typename Gyron>
class ensemble {
public:
    typedef vector<typename Gyron::state> state;
    ensemble(const vector<Gyron>& gyron, const size_t n_particles_per_lambda) : gyron_(gyron), n_particles_per_lambda_(n_particles_per_lambda) {};
    void operator()(const state& f, state& dfdx, double t) const {
        #pragma omp parallel for
        for(size_t k = 0; k < gyron_.size(); k++) {
            for(size_t j = 0; j < n_particles_per_lambda_; j++) {
                dfdx[j + k*n_particles_per_lambda_] = gyron_[k](f[j + k*n_particles_per_lambda_], t);
            }
        }
    };
    size_t size() const {
        return gyron_.size()*n_particles_per_lambda_;
    };
    size_t n_particles_per_lambda() const {
        return n_particles_per_lambda_;
    };
    const vector<Gyron>& gyron() const {
        return gyron_;
    };
private:
    const vector<Gyron> gyron_;
    const size_t n_particles_per_lambda_;
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

// Observer class to store the particle position and velocity
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
            IR3 x = particle_ensemble_.gyron()[k/particle_ensemble_.n_particles_per_lambda()].get_position(z[k]);
            temp.push_back(x[0]);
        }
        m_states.push_back(temp);
    };
private:
    const ensemble_type& particle_ensemble_;
};

// // Single particle tracing functions

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
    guiding_centre gc(Lref, Vref, charge/mass, lambda*energySI_over_refEnergy/Bref, &qsc);
    guiding_centre::state initial_state = gc.generate_state(
    {r0, theta0, phi0}, energySI_over_refEnergy,(vpp_sign > 0 ? guiding_centre::plus : guiding_centre::minus));

    vector<vector< double >> x_vec;
    boost::numeric::odeint::runge_kutta4<guiding_centre::state> integration_algorithm;
    boost::numeric::odeint::integrate_const(
        integration_algorithm, odeint_adapter(&gc),
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
//   cubic_gsl_factory ifactory;
    metric_stellna_qs_partial g(nfp, Bref, dblock_adapter(phi_grid), G0, G2, I2, iota, iotaN, B0, B1c,
                                dblock_adapter(B20), B2c, beta1s, &ifactory);
    equilibrium_stellna_qs_partial qsc(&g);
    guiding_centre gc(Lref, Vref, charge/mass, lambda*energySI_over_refEnergy/Bref, &qsc);
    guiding_centre::state initial_state = gc.generate_state(
    {r0, theta0, phi0}, energySI_over_refEnergy,(vpp_sign > 0 ? guiding_centre::plus : guiding_centre::minus));

    vector<vector< double >> x_vec;
    boost::numeric::odeint::runge_kutta4<guiding_centre::state> integration_algorithm;
    boost::numeric::odeint::integrate_const(
        integration_algorithm, odeint_adapter(&gc),
        initial_state, 0.0, Tfinal, Tfinal/nsamples, push_back_state_and_time(x_vec,&qsc,&gc) );

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
//   cubic_gsl_factory ifactory;
    metric_stellna g(field_periods, Bref, dblock_adapter(phi_grid), G0, G2, I2, iota, iotaN,
                     dblock_adapter(B0), dblock_adapter(B1c), dblock_adapter(B1s),
                     dblock_adapter(B20), dblock_adapter(B2c), dblock_adapter(B2s),
                     dblock_adapter(beta0), dblock_adapter(beta1c), dblock_adapter(beta1s),
                     &ifactory);

    equilibrium_stellna qsc(&g);

    guiding_centre gc(Lref, Vref, charge/mass, lambda*energySI_over_refEnergy/Bref, &qsc);
    guiding_centre::state initial_state = gc.generate_state(
    {r0, theta0, phi0}, energySI_over_refEnergy,(vpp_sign > 0 ? guiding_centre::plus : guiding_centre::minus));

    vector<vector< double >> x_vec;
    boost::numeric::odeint::runge_kutta4<guiding_centre::state> integration_algorithm;
    boost::numeric::odeint::integrate_const(
        integration_algorithm, odeint_adapter(&gc),
        initial_state, 0.0, Tfinal, Tfinal/nsamples, push_back_state_and_time(x_vec,&qsc,&gc) );

    return x_vec;
}

// // Ensemble particle tracing functions

// Fully quasisymmetric ensemble
vector< vector<double>> gc_solver_qs_ensemble(
                                   double G0, double G2, double I2, double nfp, double iota,
                                   double iotaN, const vector<double>& phi_grid,
                                   double B0, double B1c, double B20, double B2c,
                                   double beta1s, double charge, double mass, double energy,
                                   size_t nlambda_trapped, size_t nlambda_passing, double r0, double r_max,
                                   size_t ntheta, size_t nphi, size_t nsamples, double Tfinal, size_t nthreads
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

    std::vector<double> theta = neat_linspace(0.0, 2*numbers::pi, ntheta);
    std::vector<double> phi = neat_linspace(0.0, 2*numbers::pi/nfp, nphi);
    std::vector<double> lambda_trapped = neat_linspace(Bref/B_max, Bref/B_min, nlambda_trapped);
    std::vector<double> lambda_passing = neat_linspace(0.0, Bref/B_max*(1.0-1.0/nlambda_passing), nlambda_passing);

    vector<guiding_centre> guiding_centre_vector;
//   #pragma omp parallel for
    for(size_t k = 0; k < nlambda_trapped; k++) {
        guiding_centre_vector.push_back(guiding_centre(Lref, Vref, charge/mass, lambda_trapped[k]*energySI_over_refEnergy/Bref, &qsc));
    }
    for(size_t k = 0; k < nlambda_passing; k++) {
        guiding_centre_vector.push_back(guiding_centre(Lref, Vref, charge/mass, lambda_passing[k]*energySI_over_refEnergy/Bref, &qsc));
    }

    ensemble_type::state initial;
//   #pragma omp parallel for
    for(size_t k = 0; k < nlambda_trapped + nlambda_passing; k++) {
        for(size_t j = 0; j < ntheta; j++) {
            for(size_t l = 0; l < nphi; l++) {
                initial.push_back(guiding_centre_vector[k].generate_state(
                {r0, theta[j], phi[l]}, energySI_over_refEnergy,
                guiding_centre::vpp_sign::plus));
                initial.push_back(guiding_centre_vector[k].generate_state(
                {r0, theta[j], phi[l]}, energySI_over_refEnergy,
                guiding_centre::vpp_sign::minus));
            }
        }
    }

    vector<vector< double >> x_vec;
    ensemble_type ensemble_object(guiding_centre_vector, 2 * ntheta * nphi);
    boost::numeric::odeint::runge_kutta4<ensemble_type::state> ode_stepper;
    boost::numeric::odeint::integrate_const(
        ode_stepper, ensemble_object,
        initial, 0.0, Tfinal, Tfinal/nsamples, orbit_observer(x_vec, ensemble_object)
    );

    return x_vec;
}

// Partially quasisymmetric ensemble
vector< vector<double>> gc_solver_qs_partial_ensemble(
                                   double G0, double G2, double I2, double nfp, double iota,
                                   double iotaN, const vector<double>& phi_grid,
                                   double B0, double B1c, vector<double>& B20, double B2c,
                                   double beta1s, double charge, double mass, double energy,
                                   size_t nlambda_trapped, size_t nlambda_passing, double r0, double r_max,
                                   size_t ntheta, size_t nphi, size_t nsamples, double Tfinal, size_t nthreads
                               )
{

    omp_set_dynamic(0);
    omp_set_num_threads(nthreads);

    double Bref = B0;
    double energySI_over_refEnergy = energySIoverRefEnergy(mass, energy);
    cubic_periodic_gsl_factory ifactory;
//   cubic_gsl_factory ifactory;
    metric_stellna_qs_partial g(nfp, Bref, dblock_adapter(phi_grid), G0, G2, I2, iota, iotaN, B0, B1c,
                                dblock_adapter(B20), B2c, beta1s, &ifactory);
    equilibrium_stellna_qs_partial qsc(&g);

    double B20_max = abs(*max_element(B20.begin(), B20.end()));
    double B_max = abs(B0) + abs(r_max * B1c) + r_max * r_max * (B20_max + abs(B2c));
    double B_min = max( 0.01, abs(B0) - abs(r_max * B1c) - r_max * r_max * (B20_max + abs(B2c)) );

    std::vector<double> theta = neat_linspace(0.0, 2*numbers::pi, ntheta);
    std::vector<double> phi = neat_linspace(0.0, 2*numbers::pi/nfp, nphi);
    std::vector<double> lambda_trapped = neat_linspace(Bref/B_max, Bref/B_min, nlambda_trapped);
    std::vector<double> lambda_passing = neat_linspace(0.0, Bref/B_max*(1.0-1.0/nlambda_passing), nlambda_passing);

    vector<guiding_centre> guiding_centre_vector;
//   #pragma omp parallel for
    for(size_t k = 0; k < nlambda_trapped; k++) {
        guiding_centre_vector.push_back(guiding_centre(Lref, Vref, charge/mass, lambda_trapped[k]*energySI_over_refEnergy/Bref, &qsc));
    }
    for(size_t k = 0; k < nlambda_passing; k++) {
        guiding_centre_vector.push_back(guiding_centre(Lref, Vref, charge/mass, lambda_passing[k]*energySI_over_refEnergy/Bref, &qsc));
    }

    ensemble_type::state initial;
//   #pragma omp parallel for
    for(size_t k = 0; k < nlambda_trapped + nlambda_passing; k++) {
        for(size_t j = 0; j < ntheta; j++) {
            for(size_t l = 0; l < nphi; l++) {
                initial.push_back(guiding_centre_vector[k].generate_state(
                {r0, theta[j], phi[l]}, energySI_over_refEnergy,
                guiding_centre::vpp_sign::plus));
                initial.push_back(guiding_centre_vector[k].generate_state(
                {r0, theta[j], phi[l]}, energySI_over_refEnergy,
                guiding_centre::vpp_sign::minus));
            }
        }
    }

    vector<vector< double >> x_vec;
    ensemble_type ensemble_object(guiding_centre_vector, 2 * ntheta * nphi);
    boost::numeric::odeint::runge_kutta4<ensemble_type::state> ode_stepper;
    boost::numeric::odeint::integrate_const(
        ode_stepper, ensemble_object,
        initial, 0.0, Tfinal, Tfinal/nsamples, orbit_observer(x_vec, ensemble_object)
    );

    return x_vec;
}

// General field ensemble
vector< vector<double>> gc_solver_ensemble(
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
                                   size_t ntheta, size_t nphi, size_t nsamples, double Tfinal, size_t nthreads
                               )
{

    omp_set_dynamic(0); 
    omp_set_num_threads(nthreads);

    cubic_periodic_gsl_factory ifactory;
//   cubic_gsl_factory ifactory;
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

    std::vector<double> theta = neat_linspace(0.0, 2*numbers::pi, ntheta);
    std::vector<double> phi = neat_linspace(0.0, 2*numbers::pi/nfp, nphi);
    std::vector<double> lambda_trapped = neat_linspace(Bref/B_max, Bref/B_min, nlambda_trapped);
    std::vector<double> lambda_passing = neat_linspace(0.0, Bref/B_max*(1.0-1.0/nlambda_passing), nlambda_passing);

    vector<guiding_centre> guiding_centre_vector;
//   #pragma omp parallel for
    for(size_t k = 0; k < nlambda_trapped; k++) {
        guiding_centre_vector.push_back(guiding_centre(Lref, Vref, charge/mass, lambda_trapped[k]*energySI_over_refEnergy/Bref, &qsc));
    }
    for(size_t k = 0; k < nlambda_passing; k++) {
        guiding_centre_vector.push_back(guiding_centre(Lref, Vref, charge/mass, lambda_passing[k]*energySI_over_refEnergy/Bref, &qsc));
    }

    ensemble_type::state initial;
//   #pragma omp parallel for
    for(size_t k = 0; k < nlambda_trapped + nlambda_passing; k++) {
        for(size_t j = 0; j < ntheta; j++) {
            for(size_t l = 0; l < nphi; l++) {
                initial.push_back(guiding_centre_vector[k].generate_state(
                {r0, theta[j], phi[l]}, energySI_over_refEnergy,
                guiding_centre::vpp_sign::plus));
                initial.push_back(guiding_centre_vector[k].generate_state(
                {r0, theta[j], phi[l]}, energySI_over_refEnergy,
                guiding_centre::vpp_sign::minus));
            }
        }
    }

    vector<vector< double >> x_vec;
    ensemble_type ensemble_object(guiding_centre_vector, 2 * ntheta * nphi);
    boost::numeric::odeint::runge_kutta4<ensemble_type::state> ode_stepper;
    boost::numeric::odeint::integrate_const(
        ode_stepper, ensemble_object,
        initial, 0.0, Tfinal, Tfinal/nsamples, orbit_observer(x_vec, ensemble_object)
    );

    return x_vec;
}

#endif //NEATPP