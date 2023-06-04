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

// @equilibrium_vmec.cc, this file is part of ::gyronimo::

#include <gyronimo/core/dblock.hh>
#include "equilibrium_vmec_interp3D.hh"

namespace gyronimo{

equilibrium_vmec_interp3D::equilibrium_vmec_interp3D(
    const metric_vmec_interp3D *g, const interpolator1d_factory *ifactory)
    : IR3field_c1(abs(g->parser()->B_0()), 1.0, g), metric_(g),
      xm_nyq_(g->parser()->xm_nyq()), xn_nyq_(g->parser()->xn_nyq()),   
      bmnc_(nullptr), bsupumnc_(nullptr), bsupvmnc_(nullptr) {
  const parser_vmec *p = metric_->parser();
  dblock_adapter s_range(p->radius());
  dblock_adapter s_half_range(p->radius_half());
  // set spectral interpolators 
  bmnc_ = new interpolator1d* [xm_nyq_.size()];
  bsupumnc_ = new interpolator1d* [xm_nyq_.size()];
  bsupvmnc_ = new interpolator1d* [xm_nyq_.size()];
//@todo NEED TO FIX AXIS AND EDGE! TBI! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  #pragma omp parallel for
  for(size_t i=0; i<xm_nyq_.size(); i++) {
    std::slice s_cut (i, s_range.size(), xm_nyq_.size());
    std::valarray<double> bsupumnc_i = (p->bsupumnc())[s_cut] / this->m_factor();
    bsupumnc_[i] = ifactory->interpolate_data( s_range, dblock_adapter(bsupumnc_i));
    std::valarray<double> bsupvmnc_i = (p->bsupvmnc())[s_cut] / this->m_factor();
    bsupvmnc_[i] = ifactory->interpolate_data( s_range, dblock_adapter(bsupvmnc_i));
    // bmnc is defined a half radius
    std::slice s_h_cut (i+xm_nyq_.size(), s_half_range.size(), xm_nyq_.size());
    std::valarray<double> bmnc_i = (p->bmnc())[s_h_cut] / this->m_factor();
    bmnc_[i] = ifactory->interpolate_data( s_half_range, dblock_adapter(bmnc_i));
  };
}
equilibrium_vmec_interp3D::~equilibrium_vmec_interp3D() {
  if(bmnc_) delete bmnc_;
  if(bsupumnc_) delete bsupumnc_;
  if(bsupvmnc_) delete bsupvmnc_;
}
IR3 equilibrium_vmec_interp3D::contravariant(const IR3& position, double time) const {
  double s = position[IR3::u];
  double zeta = position[IR3::v];
  double theta = position[IR3::w];
  double B_theta = 0.0, B_zeta = 0.0;
  #pragma omp parallel for reduction(+: B_zeta, B_theta)
  for (size_t i = 0; i < xm_nyq_.size(); i++) {  
    double m = xm_nyq_[i]; double n = xn_nyq_[i];
    double cosmn = std::cos( m*theta - n*zeta );
    B_zeta += (*bsupvmnc_[i])(s) * cosmn;
    B_theta += (*bsupumnc_[i])(s) * cosmn; 
  };
  return {0.0,  B_zeta, B_theta};
}
dIR3 equilibrium_vmec_interp3D::del_contravariant(
    const IR3& position, double time) const {
  double s = position[IR3::u];
  double zeta = position[IR3::v];
  double theta = position[IR3::w];
  double B_theta = 0.0, B_zeta = 0.0;
  double dB_theta_ds = 0.0, dB_theta_dtheta = 0.0, dB_theta_dzeta = 0.0;
  double dB_zeta_ds = 0.0, dB_zeta_dtheta = 0.0, dB_zeta_dzeta = 0.0;
  #pragma omp parallel for reduction(+: B_zeta, B_theta, dB_theta_ds, dB_theta_dtheta, dB_theta_dzeta, dB_zeta_ds, dB_zeta_dtheta, dB_zeta_dzeta)
  for (size_t i = 0; i < xm_nyq_.size(); i++) {  
    double m = xm_nyq_[i]; double n = xn_nyq_[i];
    double cosmn = std::cos( m*theta - n*zeta );
    double sinmn = std::sin( m*theta - n*zeta );
    double bsupumnc_i = (*bsupumnc_[i])(s);
    double bsupvmnc_i = (*bsupvmnc_[i])(s);
    B_theta += bsupumnc_i * cosmn; 
    B_zeta += bsupvmnc_i * cosmn;
    dB_theta_ds += (*bsupumnc_[i]).derivative(s) * cosmn;
    dB_theta_dtheta -= m * bsupumnc_i * sinmn;
    dB_theta_dzeta += n * bsupumnc_i * sinmn;
    dB_zeta_ds += (*bsupvmnc_[i]).derivative(s) * cosmn;
    dB_zeta_dtheta -= m * bsupvmnc_i * sinmn;
    dB_zeta_dzeta += n * bsupvmnc_i * sinmn;
  };
  return {
      0.0, 0.0, 0.0, 
	    dB_zeta_ds, dB_zeta_dzeta, dB_zeta_dtheta,
      dB_theta_ds, dB_theta_dzeta, dB_theta_dtheta
  };
}
//@todo we can actually override the methods to calculate the covariant components of the field
//@todo move this to magnitude after the half radius issue is sorted out
double equilibrium_vmec_interp3D::magnitude_vmec(
    const IR3& position, double time) const {
  double s = position[IR3::u];
  double zeta = position[IR3::v];
  double theta = position[IR3::w];
  double Bnorm = 0.0;
  #pragma omp parallel for reduction(+: Bnorm)
  for (size_t i = 0; i < xm_nyq_.size(); i++) {  
    Bnorm += (*bmnc_[i])(s) * std::cos( xm_nyq_[i] * theta - xn_nyq_[i] *zeta );
  };
  return Bnorm;
}

}// end namespace gyronimo.
