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


#include <cmath>
#include "metric_cylindrical.hh"

namespace gyronimo {
  metric_cylindrical::metric_cylindrical(double Placeholder)
  {
    Placeholder_=Placeholder;
  }


SM3 metric_cylindrical::operator()(const IR3& r) const {
  return {1, 0.0, 0.0, r[IR3::u]*r[IR3::u], 0.0, 1};
}
dSM3 metric_cylindrical::del(const IR3& r) const {
  return {
      0.0, 0.0, 0.0, // d_i g_uu (i=u,v,w)
      0.0, 0.0, 0.0, // d_i g_uv
      0.0, 0.0, 0.0, // d_i g_uw
      2*r[IR3::u], 0.0, 0.0, //d_i g_vv
      0.0, 0.0, 0.0, // d_i g_vw
      0.0, 0.0, 0.0}; // d_i g_ww
}
double metric_cylindrical::jacobian(const IR3& r) const {
  return r[IR3::u];
}
IR3 metric_cylindrical::del_jacobian(const IR3& r) const {
  return {1, 0.0, 0.0};
}
IR3 metric_cylindrical::to_covariant(const IR3& B, const IR3& r) const {
  return {B[IR3::u], r[IR3::u]*r[IR3::u]*B[IR3::v], B[IR3::w]};
}
IR3 metric_cylindrical::to_contravariant(const IR3& B, const IR3& r) const {
  return {B[IR3::u], B[IR3::v]/(r[IR3::u]*r[IR3::u]), B[IR3::w]};
}

} // end namespace gyronimo.
