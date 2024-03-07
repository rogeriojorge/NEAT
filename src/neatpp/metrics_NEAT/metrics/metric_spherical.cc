// ::gyronimo:: - gyromotion for the people, by the people -
// An object-oriented library for gyromotion applications in plasma physics.
// Copyright (C) 2022 Paulo Rodrigues.

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

// @metric_spherical.cc, this file is part of ::gyronimo::

#include <cmath>
#include <gyronimo/metrics/metric_spherical.hh>

namespace gyronimo {

metric_spherical::metric_spherical(double radius_norm)
    : radius_norm_(radius_norm),
      radius_norm_squared_(radius_norm*radius_norm),
      radius_norm_cube_(radius_norm*radius_norm*radius_norm) {
}
SM3 metric_spherical::operator()(const IR3& r) const {
  double factor = radius_norm_squared_*r[IR3::u]*r[IR3::u];
  double sinv = std::sin(r[IR3::v]);
  return {radius_norm_squared_, 0.0, 0.0, factor, 0.0, factor*sinv*sinv};
}
dSM3 metric_spherical::del(const IR3& r) const {
  double cosv = std::cos(r[IR3::v]), sinv = std::sin(r[IR3::v]);
  double factor = 2.0*radius_norm_squared_*r[IR3::u];
  return {
      0.0, 0.0, 0.0, // d_i g_uu (i=u,v,w)
      0.0, 0.0, 0.0, // d_i g_uv
      0.0, 0.0, 0.0, // d_i g_uw
      factor, 0.0, 0.0, //d_i g_vv
      0.0, 0.0, 0.0, // d_i g_vw
      factor*sinv*sinv, factor*r[IR3::u]*sinv*cosv, 0.0}; // d_i g_ww
}
double metric_spherical::jacobian(const IR3& r) const {
  return radius_norm_cube_*r[IR3::u]*r[IR3::u]*std::sin(r[IR3::v]);
}
IR3 metric_spherical::del_jacobian(const IR3& r) const {
  double cosv = std::cos(r[IR3::v]), sinv = std::sin(r[IR3::v]);
  double factor = radius_norm_cube_*r[IR3::u];
  return {2.0*factor*sinv, factor*r[IR3::u]*cosv, 0.0};
}
IR3 metric_spherical::to_covariant(const IR3& B, const IR3& r) const {
  double factor = radius_norm_squared_*r[IR3::u]*r[IR3::u];
  double sinv = std::sin(r[IR3::v]);
  return {radius_norm_squared_*B[IR3::u],
      factor*B[IR3::v], factor*sinv*sinv*B[IR3::w]};
}
IR3 metric_spherical::to_contravariant(const IR3& B, const IR3& r) const {
  double factor = radius_norm_squared_*r[IR3::u]*r[IR3::u];
  double sinv = std::sin(r[IR3::v]);
  return {B[IR3::u]/radius_norm_squared_,
      B[IR3::v]/factor, B[IR3::w]/(factor*sinv*sinv)};
}

} // end namespace gyronimo.
