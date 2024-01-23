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

// @metric_vmec.cc, this file is part of ::gyronimo::

#include "metric_vmec_optimized.hh"
#include <gyronimo/core/error.hh>

namespace gyronimo{

metric_vmec_optimized::metric_vmec_optimized(
    const parser_vmec *p, const interpolator1d_factory *ifactory) 
    : parser_(p), b0_(p->B_0()), mnmax_(p->mnmax()), mnmax_nyq_(p->mnmax_nyq()),
      ns_(p->ns()), mpol_(p->mpol()), ntor_(p->ntor()), 
      signsgs_(p->signgs()), nfp_(p->nfp()),
      xm_(p->xm()), xn_(p->xn()), xm_nyq_(p->xm_nyq()), xn_nyq_(p->xn_nyq()),
      Rmnc_(nullptr), Zmns_(nullptr), gmnc_(nullptr)
      {
    // set radial grid block
    dblock_adapter s_range(p->radius());
    dblock_adapter s_half_range(p->radius_half());
    // set spectral components 
    Rmnc_ = new interpolator1d* [xm_.size()];
    Zmns_ = new interpolator1d* [xm_.size()];
    gmnc_ = new interpolator1d* [xm_.size()];
//@todo NEED TO FIX AXIS AND EDGE! TBI! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #pragma omp parallel for
    for(size_t i=0; i<xm_.size(); i++) {
      std::slice s_cut (i, s_range.size(), xm_.size());
      std::valarray<double> rmnc_i = (p->rmnc())[s_cut];
      Rmnc_[i] = ifactory->interpolate_data( s_range, dblock_adapter(rmnc_i));
      std::valarray<double> zmnc_i = (p->zmns())[s_cut];
      Zmns_[i] = ifactory->interpolate_data( s_range, dblock_adapter(zmnc_i));
      // note that gmnc is defined at half mesh
      std::slice s_h_cut (i+xm_nyq_.size(), s_half_range.size(), xm_nyq_.size());
      std::valarray<double> gmnc_i = (p->gmnc())[s_h_cut];
      gmnc_[i] = ifactory->interpolate_data( s_half_range, dblock_adapter(gmnc_i));
    };
}
metric_vmec_optimized::~metric_vmec_optimized() {
  if(Rmnc_) delete Rmnc_;
  if(Zmns_) delete Zmns_;
  if(gmnc_) delete gmnc_;
}
SM3 metric_vmec_optimized::operator()(const IR3& position) const {
  error(__func__, __FILE__, __LINE__, "Explicit metric not available.", 1);
  return {0, 0, 0, 0, 0, 0};
}
dSM3 metric_vmec_optimized::del(const IR3& position) const {
  error(__func__, __FILE__, __LINE__, "Explicit metric not available.", 1);
  return {
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
      0, 0, 0};
}
IR3 metric_vmec_optimized::transform2cylindrical(const IR3& position) const {
    double u = position[gyronimo::IR3::u];
    double v = position[gyronimo::IR3::v];
    double w = position[gyronimo::IR3::w];
    double R = 0.0, Z = 0.0;
  
    #pragma omp parallel for reduction(+: R, Z)
    for (size_t i = 0; i<xm_.size(); i++) {
      double m = xm_[i]; double n = xn_[i];
      R+= (*Rmnc_[i])(u) * std::cos( m*w - n*v ); 
      Z+= (*Zmns_[i])(u) * std::sin( m*w - n*v );
    }
    return  {R, v, Z};
}
 
//@todo move this to jacobian and think about testing this by calling the parent
double metric_vmec_optimized::jacobian(const IR3& position) const {
  double s = position[IR3::u];
  double zeta = position[IR3::v];
  double theta = position[IR3::w];
  double J = 0.0;
  #pragma omp parallel for reduction(+: J)
  for (size_t i = 0; i < xm_nyq_.size(); i++) {  
    J += (*gmnc_[i])(s) * std::cos( xm_nyq_[i]*theta - xn_nyq_[i]*zeta );
  };
  // left-handed VMEC coordinate system is re-oriented 
  // to u = Phi/Phi_bnd, v = zeta, w = theta for J>0
  // should we check/assume that signgs is always negative?
  return -J;
}
IR3 metric_vmec_optimized::del_jacobian(const IR3& position) const {
  double s = position[IR3::u];
  double zeta = position[IR3::v];
  double theta = position[IR3::w];
  double J_ds = 0.0, J_dzeta = 0.0, J_dtheta = 0.0;
  #pragma omp parallel for reduction(+: J_ds, J_dzeta, J_dtheta )
  for (size_t i = 0; i < xm_nyq_.size(); i++) {  
    J_ds += (*gmnc_[i]).derivative(s) * std::cos( xm_nyq_[i]*theta - xn_nyq_[i]*zeta );
    J_dzeta += xn_nyq_[i] * (*gmnc_[i])(s) * std::sin( xm_nyq_[i]*theta - xn_nyq_[i]*zeta );
    J_dtheta -= xm_nyq_[i] * (*gmnc_[i])(s) * std::sin( xm_nyq_[i]*theta - xn_nyq_[i]*zeta );
  };
  // left-handed VMEC coordinate system is re-oriented 
  // to u = Phi/Phi_bnd, v = zeta, w = theta for J>0
  // should we check/assume that signgs of derivatives is also negative?
  return {-J_ds, -J_dzeta, -J_dtheta};
}
} // end namespace gyronimo
