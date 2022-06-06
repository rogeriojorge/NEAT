// ::gyronimo:: - gyromotion for the people, by the people -
// An object-oriented library for gyromotion applications in plasma physics.
// Copyright (C) 2022 Rogerio Jorge.

// @metric_stellna_qs_partial.cc

#include <math.h>
#include <numbers>
#include "metric_stellna_qs_partial.hh"
#include <gyronimo/core/error.hh>
#include <iostream>
using namespace gyronimo;

metric_stellna_qs_partial::metric_stellna_qs_partial(
    int field_periods, double Bref, const dblock& phi_grid,
    double G0, double G2, double I2, double iota, double iotaN,
    double B0,    double B1c,
    const dblock& B20,   double B2c,
    double beta1s, const interpolator1d_factory* ifactory)
    : phi_modulus_factor_(2*std::numbers::pi/field_periods), B20_(nullptr){
  Bref_   = Bref;
  B0_     = B0;
  B1c_    = B1c;
  B20_    = ifactory->interpolate_data(phi_grid, B20);
  B2c_    = B2c;
  beta1s_ = beta1s;
  G0_     = G0;
  G2_     = G2;
  I2_     = I2;
  iota_   = iota;
  iotaN_  = iotaN;
  field_periods_ = field_periods;
}

metric_stellna_qs_partial::~metric_stellna_qs_partial() {
  if(B20_) delete B20_;
}

double metric_stellna_qs_partial::reduce_phi(double phi) const {
  phi = std::fmod(phi, phi_modulus_factor_);
  return (phi < 0 ? phi + phi_modulus_factor_ : phi);
//   return phi;
}

SM3 metric_stellna_qs_partial::operator()(const IR3& position) const {
  error(__func__, __FILE__, __LINE__, "Explicit metric not available.", 1);
  return {0, 0, 0, 0, 0, 0};
}
dSM3 metric_stellna_qs_partial::del(const IR3& position) const {
  error(__func__, __FILE__, __LINE__, "Explicit metric not available.", 1);
  return {
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
      0, 0, 0};
}

double metric_stellna_qs_partial::jacobian(const IR3& position) const {
  double r     = position[IR3::u], theta = position[IR3::v];
  double coso  = std::cos(theta),   sino = std::sin(theta);
  double cos2o = coso*coso-sino*sino;
  double phi   = this->reduce_phi(position[IR3::w]);
  double magB  = B0_+r*(B1c_*coso)+r*r*((*B20_)(phi)+B2c_*cos2o);
  double G     = G0_+r*r*G2_;
  double I     = r*r*I2_;
  double jac   = r*Bref_*(G+iota_*I)/(magB*magB);
  return jac;
}

IR3 metric_stellna_qs_partial::del_jacobian(const IR3& position) const {
  double r        = position[IR3::u], theta = position[IR3::v];
  double coso     = std::cos(theta),   sino = std::sin(theta);
  double cos2o    = coso*coso-sino*sino, sin2o = 2*coso*sino;
  double phi      = this->reduce_phi(position[IR3::w]);
  double magB     = B0_+r*( B1c_*coso)+r*r*((*B20_)(phi)+B2c_*cos2o);
  double d_u_magB =              (  B1c_*coso)+2*r*((*B20_)(phi)+B2c_*cos2o);
  double d_v_magB =            r*( -B1c_*sino)+r*r*(          -2*B2c_*sin2o);
  double d_w_magB = r*r*(*B20_).derivative(phi);
  double G        = G0_+r*r*G2_;
  double I        = r*r*I2_;
  double jac      = r*Bref_*(G+iota_*I)/(magB*magB);
  double d_u_jac  = Bref_*(G+iota_*I)/(magB*magB)+r*Bref_*(2*r*(G2_+iota_*I2_))/(magB*magB)-2*d_u_magB*jac/magB;
  double d_v_jac  = -2*d_v_magB*jac/magB;
  double d_w_jac  = -2*d_w_magB*jac/magB;
  return {d_u_jac,d_v_jac,d_w_jac};
}
