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
      Rmnc_b_(nullptr), Zmns_b_(nullptr), gmnc_b_(nullptr),
      psi_boundary_(p->phi_b()[p->phi_b().size()-1]/2/std::numbers::pi)
    {
    // set radial grid block
    dblock_adapter s_range(p->radius());
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
      std::valarray<double> gmnc_i = psi_boundary_*(p->gmnc_b())[s_cut];
      gmnc_b_[i] = ifactory->interpolate_data( s_range, dblock_adapter(gmnc_i));
    };
}
metric_boozxform::~metric_boozxform() {
  if(Rmnc_b_) delete Rmnc_b_;
  if(Zmns_b_) delete Zmns_b_;
  if(gmnc_b_) delete gmnc_b_;
}
SM3 metric_boozxform::operator()(const IR3& position) const {
  error(__func__, __FILE__, __LINE__, "Explicit metric not available.", 1);
  return {0, 0, 0, 0, 0, 0};
}
dSM3 metric_boozxform::del(const IR3& position) const {
  error(__func__, __FILE__, __LINE__, "Explicit metric not available.", 1);
  return {
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
      0, 0, 0};
}
IR3 metric_boozxform::transform2cylindrical(const IR3& position) const {
    double u = position[gyronimo::IR3::u];
    double v = std::numbers::pi-position[gyronimo::IR3::v];
    double w = position[gyronimo::IR3::w];
    double R = 0.0, Z = 0.0;
  
    #pragma omp parallel for reduction(+: R, Z)
    for (size_t i = 0; i<ixm_b_.size(); i++) {
      double m = ixm_b_[i]; double n = ixn_b_[i];
      R+= (*Rmnc_b_[i])(u) * std::cos( m*v - n*w ); 
      Z+= (*Zmns_b_[i])(u) * std::sin( m*v - n*w );
    }
    return  {R, w, Z};
}

//@todo move this to jacobian and think about testing this by calling the parent
double metric_boozxform::jacobian(const IR3& position) const {
  double s = position[IR3::u];
  double theta = std::numbers::pi-position[IR3::v];
  double zeta = position[IR3::w];
  double J = 0.0;
  #pragma omp parallel for reduction(+: J)
  // for (size_t i = 0; i < xm_nyq_.size(); i++) {  
  //   J += (*gmnc_[i])(s) * std::cos( xm_nyq_[i]*theta - xn_nyq_[i]*zeta );
  // };
  for (size_t i = 0; i < ixm_b_.size(); i++) {  
    J += (*gmnc_b_[i])(s) * std::cos( ixm_b_[i]*theta - ixn_b_[i]*zeta );
  };
  // left-handed boozxform coordinate system is re-oriented 
  // to u = Phi/Phi_bnd, v = zeta, w = theta for J>0
  // should we check/assume that signgs is always negative?
  return J;
}
IR3 metric_boozxform::del_jacobian(const IR3& position) const {
  double s = position[IR3::u];
  double theta = std::numbers::pi-position[IR3::v];
  double zeta = position[IR3::w];
  double J_ds = 0.0, J_dzeta = 0.0, J_dtheta = 0.0;
  #pragma omp parallel for reduction(+: J_ds, J_dzeta, J_dtheta )
  for (size_t i = 0; i < ixm_b_.size(); i++) {  
    J_ds     += (*gmnc_b_[i]).derivative(s)  * std::cos( ixm_b_[i]*theta - ixn_b_[i]*zeta );
    J_dtheta += ixm_b_[i] * (*gmnc_b_[i])(s) * std::sin( ixm_b_[i]*theta - ixn_b_[i]*zeta );
    J_dzeta  += ixn_b_[i] * (*gmnc_b_[i])(s) * std::sin( ixm_b_[i]*theta - ixn_b_[i]*zeta );
  };
  // left-handed VMEC coordinate system is re-oriented 
  // to u = Phi/Phi_bnd, v = zeta, w = theta for J>0
  // should we check/assume that signgs of derivatives is also negative?
  return {J_ds, J_dtheta, J_dzeta};
}
} // end namespace gyronimo
