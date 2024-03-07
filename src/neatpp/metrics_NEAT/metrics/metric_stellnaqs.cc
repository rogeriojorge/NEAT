// ::gyronimo:: - gyromotion for the people, by the people -
// An object-oriented library for gyromotion applications in plasma physics.
// Copyright (C) 2022 Rogerio Jorge and Paulo Rodrigues.

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

// @metric_stellnaqs.cc, this file is part of ::gyronimo::

#include <cmath>
#include <numbers>
#include <gyronimo/metrics/metric_stellnaqs.hh>

namespace gyronimo{

metric_stellnaqs::metric_stellnaqs(
    int field_periods, double eta_bar,
    const dblock& phi_grid, const dblock& sigma,
    const dblock& dldphi, const dblock& torsion, const dblock& curvature,
    const interpolator1d_factory* ifactory)
    : sigma_(nullptr), curvature_(nullptr), torsion_(nullptr), dldphi_(nullptr),
      phi_modulus_factor_(2*std::numbers::pi/field_periods),
      eta_bar_(eta_bar), field_periods_(field_periods) {
  sigma_ = ifactory->interpolate_data(phi_grid, sigma);
  dldphi_ = ifactory->interpolate_data(phi_grid, dldphi);
  torsion_ = ifactory->interpolate_data(phi_grid, torsion);
  curvature_ = ifactory->interpolate_data(phi_grid, curvature);
}
metric_stellnaqs::~metric_stellnaqs() {
  if(curvature_) delete curvature_;
  if(torsion_) delete torsion_;
  if(dldphi_) delete dldphi_;
  if(sigma_) delete sigma_;
}
double metric_stellnaqs::reduce_phi(double phi) const {
  return std::fmod(phi, phi_modulus_factor_);
}
SM3 metric_stellnaqs::operator()(const IR3& position) const {
  double phi = this->reduce_phi(position[IR3::w]);
  double r = position[IR3::u], theta = position[IR3::v];
  double coso = std::cos(theta), sino = std::sin(theta);
  double l_prime = (*dldphi_)(phi);
  double k = (*curvature_)(phi), k_prime = (*curvature_).derivative(phi);
  double sigma = (*sigma_)(phi), sigma_prime = (*sigma_).derivative(phi);
  double k_prime_over_k = k_prime/k;
  double eta_over_k = eta_bar_/k;
  double eta_over_k_squared = eta_over_k*eta_over_k;
  double guu = std::pow(eta_over_k*coso, 2) +
          std::pow((sino + sigma*coso)/eta_over_k, 2);
  double guv = -r*eta_over_k_squared*sino*coso +
      r/eta_over_k_squared*(sino + sigma*coso)*(coso - sigma*sino);
  double guw = r*(sigma_prime*coso*(sino + sigma*coso) +
      k_prime_over_k*(
          std::pow(sino + sigma*coso, 2) -
          std::pow(eta_over_k_squared*coso, 2)))/eta_over_k_squared;
  double gvv = r*r*(std::pow(eta_over_k*sino, 2) +
          std::pow((coso - sigma*sino)/eta_over_k, 2));
  double factor_sigma_k = sigma*k_prime_over_k + 0.5*sigma_prime;
  double gvw = r*r*(l_prime*(*torsion_)(phi) + (
          0.5*sigma_prime +
          factor_sigma_k*(coso*coso - sino*sino) +
          coso*sino*(k_prime_over_k*(1.0 - sigma*sigma) - sigma*sigma_prime)
      )/eta_over_k_squared + coso*sino*eta_over_k_squared*k_prime_over_k);
  double gww = l_prime*l_prime*(1.0 - 2.0*eta_bar_*r*coso);
  return {guu, guv, guw, gvv, gvw, gww};
}
dSM3 metric_stellnaqs::del(const IR3& position) const {
  double phi = this->reduce_phi(position[IR3::w]);
  double r = position[IR3::u], theta = position[IR3::v];
  double coso = std::cos(theta), sino = std::sin(theta);
  double l_prime = (*dldphi_)(phi), l_prime_prime = dldphi_->derivative(phi);
  double torsion = (*torsion_)(phi), torsion_prime = torsion_->derivative(phi);
  double k = (*curvature_)(phi), k_prime = (*curvature_).derivative(phi),
      k_prime_prime = (*curvature_).derivative2(phi);
  double sigma = (*sigma_)(phi), sigma_prime = (*sigma_).derivative(phi),
      sigma_prime_prime = (*sigma_).derivative2(phi);
  double eta_over_k = eta_bar_/k;
  double eta_over_k_squared = eta_over_k*eta_over_k;
  double eta_over_k_quad = eta_over_k_squared*eta_over_k_squared;
  double k_prime_over_k = k_prime/k;
  double factor_a = 1.0 + eta_over_k_quad - sigma*sigma;
  double factor_b = sigma*sigma_prime - factor_a*k_prime_over_k;
  double cos2o = coso*coso - sino*sino;
  double d_u_guu = 0.0;
  double d_v_guu = 2.0*(sino + sigma*coso)*(
      coso - sigma*sino)/eta_over_k_squared - 2.0*eta_over_k_squared*coso*sino;
  double d_w_guu =(
      2.0*sino*sino*k_prime_over_k +
      coso*sino*(4.0*sigma*k_prime_over_k + 2.0*sigma_prime) +
      coso*coso*2.0*(sigma*sigma_prime - k_prime_over_k*(
          eta_over_k_quad - sigma*sigma)))/eta_over_k_squared;
  double d_u_guv = -coso*sino*eta_over_k_squared +
      (sino + sigma*coso)*(coso - sigma*sino)/eta_over_k_squared;
  double d_v_guv = -r*(cos2o*(sigma*sigma + eta_over_k_quad - 1.0) +
          4.0*coso*sino*sigma)/eta_over_k_squared;
  double d_w_guv = r*(
      cos2o*(2.0*sigma*k_prime_over_k + sigma_prime) -
      2.0*coso*sino*factor_b)/eta_over_k_squared;
  double d_u_guw = (sino*sino*k_prime_over_k +
      coso*sino*(2.0*sigma*k_prime_over_k + sigma_prime) +
      coso*coso*(sigma*sigma_prime +
          k_prime_over_k*(sigma*sigma - eta_over_k_quad)))/eta_over_k_squared;
  double d_v_guw = r*(cos2o*(2.0*sigma*k_prime_over_k + sigma_prime) -
      2.0*coso*sino*factor_b)/eta_over_k_squared;
  double factor_k_prime2 = k_prime_over_k*k_prime_over_k + k_prime_prime/k;
  double d_w_guw = r*(
      sino*sino*factor_k_prime2 +
      coso*sino*(2.0*sigma*factor_k_prime2 + 4.0*k_prime*sigma_prime/k +
      sigma_prime_prime) + coso*coso*(
          k_prime_over_k*k_prime_over_k*(3.0*eta_over_k_quad + sigma*sigma) +
          4.0*sigma*sigma_prime*k_prime_over_k + sigma_prime*sigma_prime +
          sigma*sigma_prime_prime + k_prime_prime/k*(
              sigma*sigma - eta_over_k_quad)))/eta_over_k_squared;
  double d_u_gvv = 2.0*r*(coso*coso - 2.0*coso*sino*sigma +
      sino*sino*(eta_over_k_quad + sigma*sigma))/eta_over_k_squared;
  double d_v_gvv = 2.0*r*r*(
      coso*sino*(sigma*sigma + eta_over_k_quad - 1.0) -
      sigma*cos2o)/eta_over_k_squared;
  double d_w_gvv = 2.0*r*r*(coso*coso*k_prime_over_k -
      coso*sino*(2.0*sigma*k_prime_over_k + sigma_prime) +
      sino*sino*(sigma*sigma_prime -
          (eta_over_k_quad - sigma*sigma)*k_prime_over_k))/eta_over_k_squared;
  double d_u_gvw = 2.0*r*l_prime*torsion + r*(
      sigma_prime + cos2o*(2.0*sigma*k_prime_over_k + sigma_prime) -
      2.0*coso*sino*factor_b)/eta_over_k_squared;
  double d_v_gvw = -r*r*(cos2o*factor_b +
          2.0*coso*sino*(
              2.0*sigma*k_prime_over_k + sigma_prime))/eta_over_k_squared;
  double d_w_gvw = r*r*(eta_over_k_squared*(
          l_prime*torsion_prime + torsion*l_prime_prime) +
      0.5*sigma_prime_prime + sigma_prime*k_prime_over_k +
      cos2o*(
          sigma*(k_prime_over_k*k_prime_over_k + k_prime_prime/k) +
          2.0*sigma_prime*k_prime_over_k + 0.5*sigma_prime_prime) -
      coso*sino*(
          k_prime_over_k*k_prime_over_k*(
              sigma*sigma + 3.0*eta_over_k_quad - 1.0) +
          4.0*sigma*sigma_prime*k_prime_over_k + sigma_prime*sigma_prime +
          sigma*sigma_prime_prime - k_prime_prime/k*factor_a)
      )/eta_over_k_squared;
  double d_u_gww = -2.0*eta_bar_*l_prime*l_prime*coso;
  double d_v_gww = l_prime*l_prime*2.0*eta_bar_*r*sino;
  double d_w_gww = l_prime*l_prime_prime*(2.0 - 4.0*eta_bar_*r*coso);
  return {
      d_u_guu, d_v_guu, d_w_guu,
      d_u_guv, d_v_guv, d_w_guv,
      d_u_guw, d_v_guw, d_w_guw,
      d_u_gvv, d_v_gvv, d_w_gvv,
      d_u_gvw, d_v_gvw, d_w_gvw,
      d_u_gww, d_v_gww, d_w_gww};
}

} // end namespace gyronimo
