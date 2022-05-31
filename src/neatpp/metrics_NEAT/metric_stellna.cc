// ::gyronimo:: - gyromotion for the people, by the people -
// An object-oriented library for gyromotion applications in plasma physics.
// Copyright (C) 2021 Paulo Rodrigues, Rogerio Jorge.

// @metric_stellna.cc

#include <math.h>
#include <numbers>
#include "metric_stellna.hh"
#include <gyronimo/core/error.hh>
#include <iostream>
using namespace gyronimo;

metric_stellna::metric_stellna(int field_periods, double Bref, const dblock& phi_grid,
    double G0, double G2, double I2, double iota, double iotaN,
    const dblock& B0,    const dblock& B1c,    const dblock& B1s, 
    const dblock& B20,   const dblock& B2c,    const dblock& B2s, 
    const dblock& beta0, const dblock& beta1c, const dblock& beta1s, 
    const interpolator1d_factory* ifactory)
    : phi_modulus_factor_(2*std::numbers::pi/field_periods), B0_(nullptr),
    B1c_(nullptr), B1s_(nullptr), B20_(nullptr), B2c_(nullptr), B2s_(nullptr),
    beta0_(nullptr), beta1c_(nullptr), beta1s_(nullptr){
  Bref_   = Bref;
  B0_     = ifactory->interpolate_data(phi_grid, B0);
  B1c_    = ifactory->interpolate_data(phi_grid, B1c);
  B1s_    = ifactory->interpolate_data(phi_grid, B1s);
  B20_    = ifactory->interpolate_data(phi_grid, B20);
  B2c_    = ifactory->interpolate_data(phi_grid, B2c);
  B2s_    = ifactory->interpolate_data(phi_grid, B2s);
  beta0_  = ifactory->interpolate_data(phi_grid, beta0);
  beta1c_ = ifactory->interpolate_data(phi_grid, beta1c);
  beta1s_ = ifactory->interpolate_data(phi_grid, beta1s);
  G0_     = G0;
  G2_     = G2;
  I2_     = I2;
  iota_   = iota;
  iotaN_  = iotaN;
  field_periods_ = field_periods;
  phi0_   = phi_grid.data()[0];
}

metric_stellna::~metric_stellna() {
  if(B0_) delete B0_;
  if(B1c_) delete B1c_;
  if(B1s_) delete B1s_;
  if(B20_) delete B20_;
  if(B2c_) delete B2c_;
  if(B2s_) delete B2s_;
  if(beta0_) delete beta0_;
  if(beta1c_) delete beta1c_;
  if(beta1s_) delete beta1s_;
}

/*phi0_ added to handle phi_grid that do not start on 0
(such as the quasi-isodynamic stellarator case)*/
double metric_stellna::reduce_phi(double phi) const {
  phi = std::fmod(phi-this->phi0_, phi_modulus_factor_);
  return (phi < 0 ? phi + phi_modulus_factor_ : phi)+this->phi0_;
  // return phi;
}

SM3 metric_stellna::operator()(const IR3& position) const {
  error(__func__, __FILE__, __LINE__, "Explicit metric not available.", 1);
  return {0, 0, 0, 0, 0, 0};
}
dSM3 metric_stellna::del(const IR3& position) const {
  error(__func__, __FILE__, __LINE__, "Explicit metric not available.", 1);
  return {
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
      0, 0, 0};
}

double metric_stellna::jacobian(const IR3& position) const {
  double r     = position[IR3::u], theta = position[IR3::v];
  double coso  = std::cos(theta),   sino = std::sin(theta);
  double cos2o = coso*coso-sino*sino, sin2o = 2*coso*sino;
  double phi   = this->reduce_phi(position[IR3::w]);
  double magB  = (*B0_)(phi)+r*((*B1c_)(phi)*coso+(*B1s_)(phi)*sino)+r*r*((*B20_)(phi)+(*B2c_)(phi)*cos2o+(*B2s_)(phi)*sin2o);
  double G     = G0_+r*r*G2_;
  double I     = r*r*I2_;
  double jac   = r*Bref_*(G+iota_*I)/(magB*magB);
  return jac;
}

IR3 metric_stellna::del_jacobian(const IR3& position) const {
  double r        = position[IR3::u], theta = position[IR3::v];
  double coso     = std::cos(theta),   sino = std::sin(theta);
  double cos2o    = coso*coso-sino*sino, sin2o = 2*coso*sino;
  double phi      = this->reduce_phi(position[IR3::w]);
  double magB     = (*B0_)(phi)+r*( (*B1c_)(phi)*coso+(*B1s_)(phi)*sino)+r*r*((*B20_)(phi)+(*B2c_)(phi)*cos2o+ (*B2s_)(phi)*sin2o);
  double d_u_magB =              (  (*B1c_)(phi)*coso+(*B1s_)(phi)*sino)+2*r*((*B20_)(phi)+(*B2c_)(phi)*cos2o+ (*B2s_)(phi)*sin2o);
  double d_v_magB =            r*( -(*B1c_)(phi)*sino+(*B1s_)(phi)*coso)+r*r*(          -2*(*B2c_)(phi)*sin2o+2*(*B2s_)(phi)*cos2o);
  double d_w_magB = (*B0_).derivative(phi)+r*((*B1c_).derivative(phi)*coso+(*B1s_).derivative(phi)*sino)+r*r*((*B20_).derivative(phi)+(*B2c_).derivative(phi)*cos2o+(*B2s_).derivative(phi)*sin2o);
  double G        = G0_+r*r*G2_;
  double I        = r*r*I2_;
  double jac      = r*Bref_*(G+iota_*I)/(magB*magB);
  double d_u_jac  = Bref_*(G+iota_*I)/(magB*magB)+r*Bref_*(2*r*(G2_+iota_*I2_))/(magB*magB)-2*d_u_magB*jac/magB;
  double d_v_jac  = -2*d_v_magB*jac/magB;
  double d_w_jac  = -2*d_w_magB*jac/magB;
  return {d_u_jac,d_v_jac,d_w_jac};
}
