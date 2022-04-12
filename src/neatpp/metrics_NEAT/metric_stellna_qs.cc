// ::gyronimo:: - gyromotion for the people, by the people -
// An object-oriented library for gyromotion applications in plasma physics.
// Copyright (C) 2021 Paulo Rodrigues, Rogerio Jorge.

// @metric_stellna_qs.cc

#include <math.h>
#include <numbers>
#include "metric_stellna_qs.hh"
#include <gyronimo/core/error.hh>
#include <iostream>
using namespace gyronimo;

metric_stellna_qs::metric_stellna_qs(double Bref,
    double G0, double G2, double I2, double iota, double iotaN,
    double B0,    double B1c,
    double B20,   double B2c,
    double beta1s){
  Bref_   = Bref;
  B0_     = B0;
  B1c_    = B1c;
  B20_    = B20;
  B2c_    = B2c;
  beta1s_ = beta1s;
  G0_     = G0;
  G2_     = G2;
  I2_     = I2;
  iota_   = iota;
  iotaN_  = iotaN;
}

SM3 metric_stellna_qs::operator()(const IR3& position) const {
  error(__func__, __FILE__, __LINE__, "Explicit metric not available.", 1);
  return {0, 0, 0, 0, 0, 0};
}
dSM3 metric_stellna_qs::del(const IR3& position) const {
  error(__func__, __FILE__, __LINE__, "Explicit metric not available.", 1);
  return {
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
      0, 0, 0};
}

double metric_stellna_qs::jacobian(const IR3& position) const {
  double r     = position[IR3::u], theta = position[IR3::v];
  double coso  = std::cos(theta),   sino = std::sin(theta);
  double cos2o = coso*coso-sino*sino;
  double magB  = B0_+r*(B1c_*coso)+r*r*(B20_+B2c_*cos2o);
  double G     = G0_+r*r*G2_;
  double I     = r*r*I2_;
  double jac   = r*Bref_*(G+iota_*I)/(magB*magB);
  return jac;
}

IR3 metric_stellna_qs::del_jacobian(const IR3& position) const {
  double r        = position[IR3::u], theta = position[IR3::v];
  double coso     = std::cos(theta),   sino = std::sin(theta);
  double cos2o = coso*coso-sino*sino, sin2o = 2*coso*sino;
  double magB     = B0_+r*( B1c_*coso)+r*r*(B20_+B2c_*cos2o);
  double d_u_magB =              (  B1c_*coso)+2*r*(B20_+B2c_*cos2o);
  double d_v_magB =            r*( -B1c_*sino)+r*r*(          -2*B2c_*sin2o);
  double d_w_magB = 0;
  double G        = G0_+r*r*G2_;
  double I        = r*r*I2_;
  double jac      = r*Bref_*(G+iota_*I)/(magB*magB);
  double d_u_jac  = Bref_*(G+iota_*I)/(magB*magB)+r*Bref_*(2*r*(G2_+iota_*I2_))/(magB*magB)-2*d_u_magB*jac/magB;
  double d_v_jac  = -2*d_v_magB*jac/magB;
  double d_w_jac  = -2*d_w_magB*jac/magB;
  return {d_u_jac,d_v_jac,d_w_jac};
}
