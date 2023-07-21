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
#include "equilibrium_dommaschk.hh"
#include "metric_cylindrical.hh"
#include <gyronimo/dynamics/guiding_centre.hh>
#include <gyronimo/dynamics/odeint_adapter.hh>
#include <gyronimo/fields/linear_combo_c1.hh>
#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>
using namespace gyronimo;
using namespace std;

class push_back_state_and_time_dommaschk {
public:
  vector< vector< double > >& m_states;
  push_back_state_and_time_dommaschk(vector< vector< double > > &states,
                     const IR3field_c1* e, const guiding_centre* g)
    : m_states(states), eq_pointer_(e), gc_pointer_(g) {};
  void operator()(const guiding_centre::state& s, double t) {
    IR3 x = gc_pointer_->get_position(s);
    double B = eq_pointer_->magnitude(x, t);
    guiding_centre::state dots = (*gc_pointer_)(s, t);
    IR3 y = gc_pointer_->get_position(dots);
    double v_parallel = gc_pointer_->get_vpp(s);
    // cout << x[IR3::u]<<x[IR3::v]<<x[IR3::w] << endl;
    // cout << y[IR3::u]<<y[IR3::v]<<y[IR3::w] << endl;
    // cout << B << endl;
    // cout << v_parallel << endl;

    m_states.push_back({
        t,
        x[IR3::u], x[IR3::w], x[IR3::v],
        gc_pointer_->energy_parallel(s), 
        gc_pointer_->energy_perpendicular(s, t),
        B, v_parallel,
        y[IR3::u], y[IR3::w], y[IR3::v],
        gc_pointer_->get_vpp(dots),
        x[IR3::u], x[IR3::w], x[IR3::v]

      });
  };
private:
  const IR3field_c1* eq_pointer_;
  const guiding_centre* gc_pointer_;
};

vector< vector<double>>  dommaschktrace(
        const vector<int> m, const vector<int> l, const vector<double> coeff1, const vector<double> coeff2, const vector<double> B0,
        double charge, double mass, double Lambda,
        double vpp_sign, double energy, double R0,
        double phi0, double Z0,
        size_t nsamples, double Tfinal)
{
  double Lref = 1.0;
  double Vref = 1.0;
  double refEnergy = 0.5*codata::m_proton*mass*Vref*Vref;
  double energySI = energy*codata::e;
  double energySI_over_refEnergy = energySI/refEnergy;
  double Placeholder;
  metric_cylindrical g(Placeholder);
  int length=m.size();
  equilibrium_dommaschk deq_test_low(&g,5,2,1.4,1.4,1);
  equilibrium_dommaschk deq_test_higher(&g, 5, 4, 19.25, 0, 1);
  equilibrium_dommaschk deq_test_even_higher(&g, 5, 10, 5.1*pow(10,10),5.1*pow(10,10) , 1);

  class equilibrium_inverse_R_factor : public IR3field_c1{
    public:
      equilibrium_inverse_R_factor(const metric_cylindrical *g): IR3field_c1(1.0, 1.0, g),metric_(g){}
      virtual ~equilibrium_inverse_R_factor() override {};
      virtual IR3 contravariant(const IR3& position, double time) const override {return {0,-1/position[IR3::u],0};};
      virtual dIR3 del_contravariant(const IR3& position, double time) const override {return{0,0,0, +1/(position[IR3::u]*position[IR3::u]),0,0, 0,0,0};};
      virtual IR3 partial_t_contravariant(const IR3& position, double time) const override {return {0.0,0.0,0.0};};
    private:
      const metric_cylindrical *metric_;
  };

    class equilibrium_null_field : public IR3field_c1{
    public:
      equilibrium_null_field(const metric_cylindrical *g): IR3field_c1(1.0, 1.0, g),metric_(g){}
      virtual ~equilibrium_null_field() override {};
      virtual IR3 contravariant(const IR3& position, double time) const override {return {0,0,0};};
      virtual dIR3 del_contravariant(const IR3& position, double time) const override {return{0,0,0, 0,0,0, 0,0,0};};
      virtual IR3 partial_t_contravariant(const IR3& position, double time) const override {return {0.0,0.0,0.0};};
    private:
      const metric_cylindrical *metric_;
  };

  equilibrium_inverse_R_factor R_factor(&g);

  std::array<const IR3field_c1*, 3> p_test={&deq_test_higher,&deq_test_low,&R_factor};
  linear_combo_c1 d_test1(p_test, &g, 1, 1);
  IR3 test_position_1={R0,phi0,Z0};
  dIR3 Dmag_field=d_test1.del_contravariant(test_position_1,0);
  IR3 mag_field=d_test1.contravariant(test_position_1,0);


  if(length==1)
  {
    IR3field_c1* deq_total;
    deq_total=new equilibrium_dommaschk(&g, m[0], l[0], coeff1[0], coeff2[0], B0[0]);

  mag_field=deq_total->contravariant(test_position_1,0);
  printf("campo inserido: BR=%f,Bphi=%f,BZ=%f\n",mag_field[0],mag_field[1],mag_field[2]);

  Dmag_field=deq_total->del_contravariant(test_position_1,0);
  printf("DERIVADA EM R:BR=%f,Bphi=%f,BZ=%f\n",Dmag_field[dIR3::uu],Dmag_field[dIR3::vu],Dmag_field[dIR3::wu]);

  Dmag_field=deq_total->del_contravariant(test_position_1,0);
  printf("DERIVADA EM phi:BR=%f,Bphi=%f,BZ=%f\n",Dmag_field[dIR3::uv],Dmag_field[dIR3::vv],Dmag_field[dIR3::wv]);

  Dmag_field=deq_total->del_contravariant(test_position_1,0);
  printf("DERIVADA EM z:BR=%f,Bphi=%f,BZ=%f\n",Dmag_field[dIR3::uw],Dmag_field[dIR3::vw],Dmag_field[dIR3::ww]);

    guiding_centre gc(
        Lref, Vref, charge/mass, Lambda*energySI_over_refEnergy, deq_total);

    guiding_centre::state initial_state = gc.generate_state(
        {R0, phi0, Z0}, energySI_over_refEnergy,
        (vpp_sign > 0 ? guiding_centre::plus : guiding_centre::minus));


    cout.precision(16);
    cout.setf(ios::scientific);
    vector<vector< double >> x_vec;
    push_back_state_and_time_dommaschk observer(x_vec, deq_total, &gc);

    boost::numeric::odeint::runge_kutta4<guiding_centre::state>
        integration_algorithm;
    boost::numeric::odeint::integrate_const(
        integration_algorithm, odeint_adapter(&gc),
        initial_state, 0.0, Tfinal, Tfinal/nsamples, observer);

    return x_vec;
  }
  else
  {
    std::vector<equilibrium_dommaschk*> deq_aux;
    std::array<const IR3field_c1*, 40> p; //hardcoded to a max of 20, if linear combo c1 could receive a dblock instead of an array it would be simple to use a vecotr of undeterminate size
    const equilibrium_null_field null_field(&g);
    p.fill(&null_field);
    deq_aux.push_back(new equilibrium_dommaschk(&g, m[0], l[0], coeff1[0], coeff2[0], B0[0]));
    p[0]=deq_aux[0];  
    for(int i=1;i<length;++i)
    {
      deq_aux.push_back(new equilibrium_dommaschk(&g, m[i], l[i], coeff1[i], coeff2[i], B0[i]));
      p[i]=deq_aux[i];   
      p[40-1-i]=&R_factor;
    }
    linear_combo_c1 deq_total(p, &g, 1, 1);

  linear_combo_c1 deq_total_test2=deq_total;
  mag_field=deq_total_test2.contravariant(test_position_1,0);
  printf("campo inserido: BR=%f,Bphi=%f,BZ=%f\n",mag_field[0],mag_field[1],mag_field[2]);

  Dmag_field=deq_total_test2.del_contravariant(test_position_1,0);
  printf("DERIVADA EM R:BR=%f,Bphi=%f,BZ=%f\n",Dmag_field[dIR3::uu],Dmag_field[dIR3::vu],Dmag_field[dIR3::wu]);

 
  Dmag_field=deq_total_test2.del_contravariant(test_position_1,0);
  printf("DERIVADA EM phi:BR=%f,Bphi=%f,BZ=%f\n",Dmag_field[dIR3::uv],Dmag_field[dIR3::vv],Dmag_field[dIR3::wv]);

  Dmag_field=deq_total_test2.del_contravariant(test_position_1,0);
  printf("DERIVADA EM z:BR=%f,Bphi=%f,BZ=%f\n",Dmag_field[dIR3::uw],Dmag_field[dIR3::vw],Dmag_field[dIR3::ww]);


  guiding_centre gc(
      Lref, Vref, charge/mass, Lambda*energySI_over_refEnergy, &deq_total);

  guiding_centre::state initial_state = gc.generate_state(
      {R0, phi0, Z0}, energySI_over_refEnergy,
      (vpp_sign > 0 ? guiding_centre::plus : guiding_centre::minus));

  cout.precision(16);
  cout.setf(ios::scientific);
  vector<vector< double >> x_vec;
  push_back_state_and_time_dommaschk observer(x_vec, &deq_total, &gc);

  boost::numeric::odeint::runge_kutta4<guiding_centre::state>
      integration_algorithm;
  boost::numeric::odeint::integrate_const(
      integration_algorithm, odeint_adapter(&gc),
      initial_state, 0.0, Tfinal, Tfinal/nsamples, observer);

  return x_vec;
  }
}
