// ::gyronimo:: - gyromotion for the people, by the people -
// An object-oriented library for gyromotion applications in plasma physics.
// Copyright (C) 2022 Jorge Ferreira and Paulo Rodrigues.

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

// @metric_boozxform.cc, this file is part of ::gyronimo::

#include "metric_boozxform.hh"
#include <gyronimo/core/error.hh>

namespace gyronimo{

metric_boozxform::metric_boozxform(
    const parser_boozxform *p, const interpolator1d_factory *ifactory) 
    : parser_(p), b0_(p->B_0()), mnboz_b_(p->mnboz_b()),
      ns_b_(p->ns_b()), ixm_b_(p->ixm_b()), ixn_b_(p->ixn_b()), 
      Rmnc_b_(nullptr), Zmns_b_(nullptr), gmnc_b_(nullptr)
    {
    // set radial grid block
    dblock_adapter s_range(p->radius());
    // dblock_adapter s_half_range(p->radius_half());
    // set spectral components 
    Rmnc_b_ = new interpolator1d* [ixm_b_.size()];
    Zmns_b_ = new interpolator1d* [ixm_b_.size()];
    gmnc_b_ = new interpolator1d* [ixm_b_.size()];
//@todo NEED TO FIX AXIS AND EDGE! TBI! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #pragma omp parallel for
    for(size_t i=0; i<ixm_b_.size(); i++) {
      std::slice s_cut (i, s_range.size(), ixm_b_.size());
      std::valarray<double> rmnc_i = (p->rmnc_b())[s_cut];
      Rmnc_b_[i] = ifactory->interpolate_data( s_range, dblock_adapter(rmnc_i));
      std::valarray<double> zmnc_i = (p->zmns_b())[s_cut];
      Zmns_b_[i] = ifactory->interpolate_data( s_range, dblock_adapter(zmnc_i));
      // note that gmnc is defined at half mesh
      // std::slice s_h_cut (i+xm_nyq_.size(), s_half_range.size(), xm_nyq_.size());
      // std::valarray<double> gmnc_i = (p->gmnc_b())[s_h_cut];
      std::valarray<double> gmnc_i = (p->gmnc_b())[s_cut];
      gmnc_b_[i] = ifactory->interpolate_data( s_range, dblock_adapter(gmnc_i));
    };
}
metric_boozxform::~metric_boozxform() {
  if(Rmnc_b_) delete Rmnc_b_;
  if(Zmns_b_) delete Zmns_b_;
  if(gmnc_b_) delete gmnc_b_;
}
SM3 metric_boozxform::operator()(const IR3& position) const {
  double s = position[IR3::u];
  double zeta = position[IR3::v];
  double theta = position[IR3::w];
  double R = 0.0, dR_ds = 0.0, dR_dtheta = 0.0, dR_dzeta = 0.0;
  double Z = 0.0, dZ_ds = 0.0, dZ_dtheta = 0.0, dZ_dzeta = 0.0;

  #pragma omp parallel for reduction(+: R, Z, dR_ds, dR_dtheta, dR_dzeta, dZ_ds, dZ_dtheta, dZ_dzeta)
  for (size_t i = 0; i<ixm_b_.size(); i++) {  
    double m = ixm_b_[i]; double n = ixn_b_[i];
    double cosmn = std::cos( m*theta - n*zeta );
    double sinmn = std::sin( m*theta - n*zeta );
    double rmnc_i = (*Rmnc_b_[i])(s); 
    double zmns_i = (*Zmns_b_[i])(s);
    // assuming for now that boozxform equilibrium has stellarator symmetry.
    R += rmnc_i * cosmn; 
    Z += zmns_i * sinmn;
    dR_ds += (*Rmnc_b_[i]).derivative(s) * cosmn; 
    dR_dtheta -= m * rmnc_i * sinmn; 
    dR_dzeta += n * rmnc_i * sinmn;
    dZ_ds += (*Zmns_b_[i]).derivative(s) * sinmn; 
    dZ_dtheta += m * zmns_i * cosmn; 
    dZ_dzeta -= n * zmns_i * cosmn; 
  };
  return {
    dR_ds * dR_ds + dZ_ds * dZ_ds,                      // g_uu
    dR_ds * dR_dzeta + dZ_ds * dZ_dzeta,                // g_uw
    dR_ds * dR_dtheta + dZ_ds * dZ_dtheta,              // g_uv
    R * R + dR_dzeta * dR_dzeta + dZ_dzeta * dZ_dzeta,  // g_vv
    dR_dtheta * dR_dzeta + dZ_dtheta * dZ_dzeta,        // g_vw
    dR_dtheta * dR_dtheta + dZ_dtheta * dZ_dtheta       // g_ww
  };
}
dSM3 metric_boozxform::del(const IR3& position) const {
  double s = position[IR3::u];
  double zeta = position[IR3::v];
  double theta = position[IR3::w];
  double R = 0.0, Z = 0.0;
  double dR_ds = 0.0,        dR_dtheta = 0.0,       dR_dzeta = 0.0;
  double d2R_ds2 = 0.0,      d2R_dsdtheta = 0.0,    d2R_dsdzeta = 0.0;
  double d2R_dthetads = 0.0, d2R_dtheta2 = 0.0,     d2R_dthetadzeta = 0.0; 
  double d2R_dzetads = 0.0,  d2R_dzetadtheta = 0.0, d2R_dzeta2 = 0.0;
  double dZ_ds = 0.0,        dZ_dtheta = 0.0,       dZ_dzeta = 0.0;
  double d2Z_ds2 = 0.0,      d2Z_dsdtheta = 0.0,    d2Z_dsdzeta = 0.0;
  double d2Z_dthetads = 0.0, d2Z_dtheta2 = 0.0,     d2Z_dthetadzeta = 0.0; 
  double d2Z_dzetads = 0.0,  d2Z_dzetadtheta = 0.0, d2Z_dzeta2 = 0.0;

  #pragma omp parallel for reduction(+: R, dR_ds, dR_dtheta, dR_dzeta, d2R_ds2, d2R_dsdtheta, d2R_dsdzeta, d2R_dtheta2, d2R_dthetadzeta, d2R_dzeta2, Z, dZ_ds ,dZ_dtheta, dZ_dzeta, d2Z_ds2, d2Z_dsdtheta, d2Z_dsdzeta, d2Z_dtheta2, d2Z_dthetadzeta, d2Z_dzeta2)
  for (size_t i = 0; i<ixm_b_.size(); i++) {  
    double m = ixm_b_[i]; double n = ixn_b_[i];
    double cosmn = std::cos( m*theta - n*zeta );
    double sinmn = std::sin( m*theta - n*zeta );
    double rmnc_i = (*Rmnc_b_[i])(s); 
    double zmns_i = (*Zmns_b_[i])(s);
    double d_rmnc_i = (*Rmnc_b_[i]).derivative(s); 
    double d_zmns_i = (*Zmns_b_[i]).derivative(s); 
    double d2_rmnc_i = (*Rmnc_b_[i]).derivative2(s);
    double d2_zmns_i = (*Zmns_b_[i]).derivative2(s);
    // assuming for now that boozxform equilibrium has stellarator symmetry.
    R += rmnc_i * cosmn; Z += zmns_i * sinmn;
    dR_ds += d_rmnc_i * cosmn; 
    dR_dtheta -= m * rmnc_i * sinmn; 
    dR_dzeta += n * rmnc_i * sinmn;
    d2R_ds2 += d2_rmnc_i * cosmn; 
    d2R_dsdtheta -= m * d_rmnc_i * sinmn;
    d2R_dsdzeta += n * d_rmnc_i * sinmn;
    d2R_dtheta2 -= m * m * rmnc_i * cosmn;
    d2R_dthetadzeta += m * n * rmnc_i * cosmn;
    d2R_dzeta2 -= n * n * rmnc_i * cosmn;
    dZ_ds += d_zmns_i * sinmn; 
    dZ_dtheta += m * zmns_i * cosmn; 
    dZ_dzeta -= n * zmns_i * cosmn; 
    d2Z_ds2 += d2_zmns_i * sinmn;
    d2Z_dsdtheta += m * d_zmns_i * cosmn;
    d2Z_dsdzeta -= n * d_zmns_i * cosmn;
    d2Z_dtheta2 -= m * m * zmns_i * sinmn;
    d2Z_dthetadzeta += m * n * zmns_i * sinmn;
    d2Z_dzeta2 -= n * n * zmns_i * sinmn;
}
//@todo still need to test this carefully. Find a way to test d_g!
  return {
      2 * (dR_ds * d2R_ds2      + dZ_ds * d2Z_ds2), 
      2 * (dR_ds * d2R_dsdzeta  + dZ_ds * d2Z_dsdzeta), // d_i g_uu
      2 * (dR_ds * d2R_dsdtheta + dZ_ds * d2Z_dsdtheta), 
      dR_ds * d2R_dsdzeta       + dR_dzeta * d2R_ds2      + dZ_ds * d2Z_dsdzeta      + dZ_dzeta * d2Z_ds2,
      dR_ds * d2R_dzeta2        + dR_dzeta * d2R_dsdzeta  + dZ_ds * d2Z_dzeta2       + dZ_dzeta * d2Z_dsdzeta,// d_i g_uv
      dR_ds * d2R_dthetadzeta   + dR_dzeta * d2R_dsdtheta  + dZ_ds * d2Z_dthetadzeta  + dZ_dzeta * d2Z_dsdtheta, 
      dR_ds * d2R_dsdtheta      + dR_dtheta * d2R_ds2      + dZ_ds * d2Z_dsdtheta     + dZ_dtheta * d2Z_ds2,
      dR_ds * d2R_dthetadzeta   + dR_dtheta * d2R_dsdzeta  + dZ_ds * d2Z_dthetadzeta  + dZ_dtheta * d2Z_dsdzeta, // d_i g_uw
      dR_ds * d2R_dtheta2       + dR_dtheta * d2R_dsdtheta + dZ_ds * d2Z_dtheta2      + dZ_dtheta * d2Z_dsdtheta, 
      2 * (R * dR_ds     + dR_dzeta * d2R_dsdzeta     + dZ_dzeta * d2Z_dsdzeta), 
      2 * (R * dR_dzeta  + dR_dzeta * d2R_dzeta2      + dZ_dzeta * d2Z_dzeta2),  // d_i g_vv
      2 * (R * dR_dtheta + dR_dzeta * d2R_dthetadzeta + dZ_dzeta * d2Z_dthetadzeta),  
      dR_dtheta * d2R_dsdzeta     + dR_dzeta * d2R_dsdtheta     + dZ_dtheta * d2Z_dsdzeta      + dZ_dzeta * d2Z_dsdtheta,
      dR_dtheta * d2R_dzeta2      + dR_dzeta * d2R_dthetadzeta  + dZ_dtheta * d2Z_dzeta2       + dZ_dzeta * d2Z_dthetadzeta, // d_i g_vw
      dR_dtheta * d2R_dthetadzeta + dR_dzeta * d2R_dtheta2      + dZ_dtheta * d2Z_dthetadzeta  + dZ_dzeta * d2Z_dtheta2,  
      2 * (dR_dtheta * d2R_dsdtheta     + dZ_dtheta * d2Z_dsdtheta), 
      2 * (dR_dtheta * d2R_dthetadzeta  + dZ_dtheta * d2Z_dthetadzeta), // d_i g_ww
      2 * (dR_dtheta * d2R_dtheta2      + dZ_dtheta * d2Z_dtheta2),  
  };
}
IR3 metric_boozxform::transform2cylindrical(const IR3& position) const {
    double u = position[gyronimo::IR3::u];
    double v = position[gyronimo::IR3::v];
    double w = position[gyronimo::IR3::w];
    double R = 0.0, Z = 0.0;
  
    #pragma omp parallel for reduction(+: R, Z)
    for (size_t i = 0; i<ixm_b_.size(); i++) {
      double m = ixm_b_[i]; double n = ixn_b_[i];
      R+= (*Rmnc_b_[i])(u) * std::cos( m*w - n*v ); 
      Z+= (*Zmns_b_[i])(u) * std::sin( m*w - n*v );
    }
    return  {R, v, Z};
}
 
//@todo move this to jacobian and think about testing this by calling the parent
// double metric_boozxform::jacobian(const IR3& position) const {
//   double s = position[IR3::u];
//   double zeta = position[IR3::v];
//   double theta = position[IR3::w];
//   double J = 0.0;
//   #pragma omp parallel for reduction(+: J)
//   // for (size_t i = 0; i < xm_nyq_.size(); i++) {  
//   //   J += (*gmnc_[i])(s) * std::cos( xm_nyq_[i]*theta - xn_nyq_[i]*zeta );
//   // };
//   for (size_t i = 0; i < ixm_b_.size(); i++) {  
//     J += (*gmnc_b_[i])(s) * std::cos( ixm_b_[i]*theta - ixn_b_[i]*zeta );
//   };
//   // left-handed boozxform coordinate system is re-oriented 
//   // to u = Phi/Phi_bnd, v = zeta, w = theta for J>0
//   // should we check/assume that signgs is always negative?
//   return J;
// }
// IR3 metric_boozxform::del_jacobian(const IR3& position) const {
//   double s = position[IR3::u];
//   double zeta = position[IR3::v];
//   double theta = position[IR3::w];
//   double J_ds = 0.0, J_dzeta = 0.0, J_dtheta = 0.0;
//   #pragma omp parallel for reduction(+: J_ds, J_dzeta, J_dtheta )
//   for (size_t i = 0; i < ixm_b_.size(); i++) {  
//     J_ds += (*gmnc_b_[i]).derivative(s) * std::cos( ixm_b_[i]*theta - ixn_b_[i]*zeta );
//     J_dzeta += ixn_b_[i] * (*gmnc_b_[i])(s) * std::sin( ixm_b_[i]*theta - ixn_b_[i]*zeta );
//     J_dtheta -= ixm_b_[i] * (*gmnc_b_[i])(s) * std::sin( ixm_b_[i]*theta - ixn_b_[i]*zeta );
//   };
//   // left-handed boozxform coordinate system is re-oriented 
//   // to u = Phi/Phi_bnd, v = zeta, w = theta for J>0
//   // should we check/assume that signgs of derivatives is also negative?
//   return {-J_ds, -J_dzeta, -J_dtheta};
// }
} // end namespace gyronimo
