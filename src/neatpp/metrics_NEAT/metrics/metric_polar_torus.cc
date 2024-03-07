// ::gyronimo:: - gyromotion for the people, by the people -
// An object-oriented library for gyromotion applications in plasma physics.
// Copyright (C) 2021 Paulo Rodrigues.

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

// @metric_polar_torus.cc, this file is part of ::gyronimo::

#include <cmath>
#include <gyronimo/metrics/metric_polar_torus.hh>

namespace gyronimo {

metric_polar_torus::metric_polar_torus(
    const double minor_radius, double major_radius)
    : minor_radius_(minor_radius), major_radius_(major_radius),
      minor_radius_squared_(minor_radius_*minor_radius_),
      major_radius_squared_(major_radius_*major_radius_),
      iaspect_ratio_(minor_radius_/major_radius_) {
}
SM3 metric_polar_torus::operator()(const IR3& r) const {
  double u = r[IR3::u], v = r[IR3::v];
  double R = major_radius_*(1.0 + iaspect_ratio_*u*std::cos(v));
  return {minor_radius_squared_, 0.0, 0.0,
           minor_radius_squared_*u*u, 0.0,
                                      R*R};
}
dSM3 metric_polar_torus::del(const IR3& r) const {
  double u = r[IR3::u];
  double v = r[IR3::v];
  double cosv = std::cos(v), sinv = std::sin(v);
  double R = major_radius_*(1.0 + iaspect_ratio_*u*cosv);
  double factor = 2.0*R*minor_radius_;
  return {
      0.0, 0.0, 0.0, // d_i g_uu
      0.0, 0.0, 0.0, // d_i g_uv
      0.0, 0.0, 0.0, // d_i g_uw
      2.0 * u * minor_radius_squared_, 0.0, 0.0, //d_i g_vv
      0.0, 0.0, 0.0, // d_i g_vw
      factor * cosv, - factor * u * sinv, 0.0}; // d_i g_ww
}
double metric_polar_torus::jacobian(const IR3& r) const {
  double u = r[IR3::u], v = r[IR3::v];
  return minor_radius_squared_*major_radius_*
      u*(1.0 + iaspect_ratio_*u*std::cos(v));
}
IR3 metric_polar_torus::to_covariant(const IR3& B, const IR3& r) const {
  double u = r[IR3::u];
  double v = r[IR3::v];
  return {
    minor_radius_squared_*B[IR3::u],
    minor_radius_squared_*u*u*B[IR3::v],
    std::pow(major_radius_*(1.0 + iaspect_ratio_*u*std::cos(v)), 2)*B[IR3::w]};
}
IR3 metric_polar_torus::to_contravariant(const IR3& B, const IR3& r) const {
  double u = r[IR3::u];
  double v = r[IR3::v];
  return {
    B[IR3::u]/minor_radius_squared_,
    B[IR3::v]/(minor_radius_squared_*u*u),
    B[IR3::w]/std::pow(major_radius_*(1.0 + iaspect_ratio_*u*std::cos(v)), 2)};
}

} // end namespace gyronimo.
