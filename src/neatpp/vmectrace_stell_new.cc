// ::gyronimo:: - gyromotion for the people, by the people -
// An object-oriented library for gyromotion applications in plasma physics.
// Copyright (C) 2021 Paulo Rodrigues.

// ::gyronimo:: is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// ::gyronimo:: is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with ::gyronimo::.  If not, see <https://www.gnu.org/licenses/>.

// @vmectrace.cc, this file is part of ::gyronimo::

// Command-line tool to print guiding-centre orbits in `VMEC` equilibria.
// External dependencies:
// - [argh](https://github.com/adishavit/argh), a minimalist argument handler.
// - [GSL](https://www.gnu.org/software/gsl), the GNU Scientific Library.
// - [boost](https://www.gnu.org/software/gsl), the boost library.

#ifdef OPENMP
#include <omp.h>
#endif
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
#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>


// ODEInt observer object to insert state in x_vec at each time step.

class orbit_observer_new {
public:
  std::vector< std::vector< double > >& m_states;
  orbit_observer_new(
      /*double zstar, double vstar,*/ 
      std::vector< std::vector< double > > &states,
      const gyronimo::IR3field_c1* e, const gyronimo::guiding_centre* g)
    : /*zstar_(zstar), vstar_(vstar),*/ m_states(states), eq_pointer_(e), gc_pointer_(g) {};
  void operator()(const gyronimo::guiding_centre::state& s, double t) {
    gyronimo::IR3 x = gc_pointer_->get_position(s);
    double v_parallel = gc_pointer_->get_vpp(s);
    double B = eq_pointer_->magnitude(x, t);
    //double bphi = eq_pointer_->covariant_versor(x, t)[gyronimo::IR3::w];
    //double flux = x[gyronimo::IR3::u]*x[gyronimo::IR3::u];
    gyronimo::IR3 X = eq_pointer_->metric()->transform2cylindrical(x);
    gyronimo::guiding_centre::state dots = (*gc_pointer_)(s, t);
    gyronimo::IR3 y = gc_pointer_->get_position(dots);
    m_states.push_back({
        t, x[gyronimo::IR3::u], x[gyronimo::IR3::v], x[gyronimo::IR3::w],
        X[gyronimo::IR3::u], X[gyronimo::IR3::v], X[gyronimo::IR3::w],
        gc_pointer_->energy_parallel(s), 
        gc_pointer_->energy_perpendicular(s, t), B, v_parallel,
        y[gyronimo::IR3::u], y[gyronimo::IR3::v], y[gyronimo::IR3::w],
        gc_pointer_->get_vpp(dots)
      });
  };
private:
  double zstar_, vstar_;
  const gyronimo::IR3field_c1* eq_pointer_;
  const gyronimo::guiding_centre* gc_pointer_;
};

/*
   The canonical toroidal momentum is
     P_phi = q_s A_\phi + m_s v_\parallel B_\phi/B,
   where
     \mathbf{B} = \nabla \phi \times \nabla \Psi + B_\phi \nabla \phi,
     \mathbf{B} = \nabla \times \mathbf{A} => A_\phi = - \Psi,
   and all variables are in SI units, except \Phi and \Psi in Wb/rad. Writting P_\phi in
   eV.s and the energy E in eV, with e the electron charge, one gets:
     P_phi = - (q_s/e) \Psi +
         \sigma_\parallel b_\phi \sqrt{2 m_s (E/e) \Lambda (1 - B/B_0)}
   Function arguments:
   pphi: \P_\phi in eV.s;
   zstar: (q_s/e) \Psi_b, with \Psi_b the boundary flux, q_s the species charge;
   vdagger: \sqrt{2 m_s (E/e)}, with E in eV, m_s the species mass;
   lambda: \Lambda;
   vpp_sign: the name says it all;
   veq: reference to an VMEC equilibrium object;
*/

/*
Important info:
    mass   - m_proton units.
    charge - q_electron units.
    energy - eV.
    s      - initial normalized radial position/surface (0-1) , s = \sqrt{\Phi/\Phi_b}
    lambda - lambda = mu/Energy = v_perp^2/(v_parallel^2+v_\perp^2) (signed)
*/

std::vector< std::vector<double>>  vmectrace_stell(double mass=1.0,double charge=1.0,double energy=1.0,double s=0.1,
                    double th=0,double phi=0,double lambda=1.0,double Tfinal=1.0, std::size_t nsamples=1024, const std::string& map="output.txt") {

  gyronimo::parser_vmec vmap(map);
  gyronimo::cubic_gsl_factory ifactory;
  gyronimo::metric_vmec g(&vmap, &ifactory);
  gyronimo::equilibrium_vmec veq(&g, &ifactory);
  
  double vpp_sign = std::copysign(1.0, lambda);  // lambda carries vpp sign.
  lambda = std::abs(lambda);  // once vpp sign is stored, lambda turns unsigned.

// Computes normalisation constants:
  double Vref = 1; // New version of Valfven
  double Uref = 0.5*gyronimo::codata::m_proton*mass*Vref*Vref; // New version of Ualfven
  double energySI = energy*gyronimo::codata::e;
  double Lref= 1.0;

// Builds the guiding_centre object:
  gyronimo::guiding_centre gc(
      Lref, Vref, charge/mass, lambda*energySI/Uref, &veq);

// Computes the initial conditions from the supplied constants of motion:
  /*double zstar = charge*g.parser()->cpsurf()*veq.B_0()*veq.R_0()*veq.R_0();
  double vstar = Vref*mass*gyronimo::codata::m_proton/gyronimo::codata::e;*/
  //double vdagger = vstar*std::sqrt(energySI/Uref);
  gyronimo::guiding_centre::state initial_state = gc.generate_state(
      {s , th, phi}, energySI/Uref,
      (vpp_sign > 0 ?
        gyronimo::guiding_centre::plus : gyronimo::guiding_centre::minus));

// integrates for t in [0,Tfinal], with dt=Tfinal/nsamples, using RK4.
  std::cout.precision(16);
  std::cout.setf(std::ios::scientific);
  std::vector<std::vector< double >> x_vec;
  orbit_observer_new observer(/*zstar, vstar,*/ x_vec, &veq, &gc);

  boost::numeric::odeint::runge_kutta4<gyronimo::guiding_centre::state>
      integration_algorithm;
  boost::numeric::odeint::integrate_const(
      integration_algorithm, gyronimo::odeint_adapter(&gc),
      initial_state, 0.0, Tfinal, Tfinal/nsamples, observer);

  return x_vec;

}

// Python wrapper functions
PYBIND11_MODULE(vmectrace_stell, m) {
    m.doc() = "Gyronimo Wrapper for Vmectrace_stell";
    m.def("vmectrace_stell",&vmectrace_stell);
}
