// ::gyronimo:: - gyromotion for the people, by the people -
// An object-oriented library for gyromotion applications in plasma physics.
// Copyright (C) 2021 Paulo Rodrigues, Rogerio Jorge.

// @equilibrium_stellna.cc

#include <numbers>
#include "equilibrium_stellna.hh"
#include <valarray>
using namespace gyronimo;

equilibrium_stellna::equilibrium_stellna(
    const metric_stellna *g)
    : IR3field_c1(g->Bref(), 1.0, g), metric_(g){
}

IR3 equilibrium_stellna::contravariant(const IR3& position, double time) const {
  double r    = position[IR3::u];
  double jac   = metric_->jacobian(position);
//   double Bu    = 0;
//   double Bv    = r*metric_->Bref()*metric_->iotaN()/jac;
  double Bw    = r*metric_->Bref()/jac;
  return {0, metric_->iotaN()*Bw, Bw};
}

dIR3 equilibrium_stellna::del_contravariant(const IR3& position, double time) const {
  double r = position[IR3::u];
  double jac = metric_->jacobian(position);
  double d_u_jac = metric_->del_jacobian(position)[IR3::u];
  double d_v_jac = metric_->del_jacobian(position)[IR3::v];
  double d_w_jac = metric_->del_jacobian(position)[IR3::w];

//   double d_u_Bu = 0;
//   double d_v_Bu = 0;
//   double d_w_Bu = 0;

//   double d_u_Bv = metric_->Bref()*metric_->iotaN()/jac-d_u_jac*r*metric_->Bref()*metric_->iotaN()/(jac*jac);
//   double d_v_Bv = -d_v_jac*r*metric_->Bref()*metric_->iotaN()/(jac*jac);
//   double d_w_Bv = -d_w_jac*r*metric_->Bref()*metric_->iotaN()/(jac*jac);

  double d_u_Bw = metric_->Bref()/jac-d_u_jac*r*metric_->Bref()/(jac*jac);
  double d_v_Bw = -d_v_jac*r*metric_->Bref()/(jac*jac);
  double d_w_Bw = -d_w_jac*r*metric_->Bref()/(jac*jac);

  return {
      0, 0, 0,
      metric_->iotaN()*d_u_Bw, metric_->iotaN()*d_v_Bw, metric_->iotaN()*d_w_Bw,
      d_u_Bw, d_v_Bw, d_w_Bw};
}

IR3 equilibrium_stellna::covariant(const IR3& position, double time) const {
  double r = position[IR3::u], theta = position[IR3::v];
  double phi = metric_->reduce_phi(position[IR3::w]);
  double coso  = std::cos(theta),   sino = std::sin(theta);
  double Bu = r*metric_->Bref()*((*metric_->beta0())(phi)+r*((*metric_->beta1c())(phi)*coso+(*metric_->beta1s())(phi)*sino));
  double Bv = r*r*metric_->I2();
  double Bw = metric_->G0()+r*r*( metric_->G2()+(metric_->iota()-metric_->iotaN())*metric_->I2());
  return {Bu, Bv, Bw};
}

dIR3 equilibrium_stellna::del_covariant(const IR3& position, double time) const {
  double r = position[IR3::u], theta = position[IR3::v];
  double phi = metric_->reduce_phi(position[IR3::w]);
  double coso  = std::cos(theta),   sino = std::sin(theta);

  double d_u_Bu =   metric_->Bref()*((*metric_->beta0())(phi)+2*r*( (*metric_->beta1c())(phi)*coso+(*metric_->beta1s())(phi)*sino));
  double d_v_Bu = r*metric_->Bref()*(                 r*(-(*metric_->beta1c())(phi)*sino+(*metric_->beta1s())(phi)*coso));
  double d_w_Bu = r*metric_->Bref()*((*metric_->beta0()).derivative(phi)+r*((*metric_->beta1c()).derivative(phi)*coso+(*metric_->beta1s()).derivative(phi)*sino));

  double d_u_Bv = 2*r*metric_->I2();
  double d_v_Bv = 0;
  double d_w_Bv = 0;

  double d_u_Bw = 2*r*( metric_->G2()+(metric_->iota()-metric_->iotaN())*metric_->I2());
  double d_v_Bw = 0;
  double d_w_Bw = 0;

  return {
      d_u_Bu, d_v_Bu, d_w_Bu,
      d_u_Bv, d_v_Bv, d_w_Bv,
      d_u_Bw, d_v_Bw, d_w_Bw};
}

double equilibrium_stellna::magnitude(const IR3& position, double time) const {
  double r = position[IR3::u], theta = position[IR3::v];
  double phi = metric_->reduce_phi(position[IR3::w]);
  double coso  = std::cos(theta),   sino = std::sin(theta);
  double cos2o = 2*coso*coso-1, sin2o = 2*sino*coso;
  double magB  = (*metric_->B0())(phi)+r*((*metric_->B1c())(phi)*coso+(*metric_->B1s())(phi)*sino)+r*r*((*metric_->B20())(phi)+(*metric_->B2c())(phi)*cos2o+(*metric_->B2s())(phi)*sin2o);
  return magB;
}

IR3 equilibrium_stellna::del_magnitude(const IR3& position, double time) const {
  double r = position[IR3::u], theta = position[IR3::v];
  double phi = metric_->reduce_phi(position[IR3::w]);
  double coso  = std::cos(theta),   sino = std::sin(theta);
  double cos2o = 2*coso*coso-1, sin2o = 2*sino*coso;
  double d_u_magB =   ( (*metric_->B1c())(phi)*coso+(*metric_->B1s())(phi)*sino)+2*r*((*metric_->B20())(phi)+(*metric_->B2c())(phi)*cos2o+  (*metric_->B2s())(phi)*sin2o);
  double d_v_magB = r*(-(*metric_->B1c())(phi)*sino+(*metric_->B1s())(phi)*coso)+r*r*(          -2*(*metric_->B2c())(phi)*sin2o+2*(*metric_->B2s())(phi)*cos2o);
  double d_w_magB = (*metric_->B0()).derivative(phi)+r*((*metric_->B1c()).derivative(phi)*coso+(*metric_->B1s()).derivative(phi)*sino)+r*r*((*metric_->B20()).derivative(phi)+(*metric_->B2c()).derivative(phi)*cos2o+(*metric_->B2s()).derivative(phi)*sin2o);
  return {d_u_magB,d_v_magB,d_w_magB};
}
