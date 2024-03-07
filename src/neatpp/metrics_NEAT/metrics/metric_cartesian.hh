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

// @metric_cartesian.hh, this file is part of ::gyronimo::

#ifndef GYRONIMO_METRIC_CARTESIAN
#define GYRONIMO_METRIC_CARTESIAN

#include <gyronimo/metrics/metric_covariant.hh>

namespace gyronimo {

//! Trivial covariant metric for cartesian space.
class metric_cartesian : public metric_covariant {
 public:
  metric_cartesian() {};
  virtual ~metric_cartesian() override {};

  virtual SM3 operator()(const IR3& r) const override {
    return {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
  };
  virtual dSM3 del(const IR3& r) const override {
    return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  };
  virtual double jacobian(const IR3& r) const override {
    return 1.0;
  };
  virtual IR3 del_jacobian(const IR3& r) const override {
    return {0.0, 0.0, 0.0};
  };
  virtual IR3 to_covariant(const IR3& B, const IR3& r) const override {
    return B;
  };
  virtual IR3 to_contravariant(const IR3& B, const IR3& r) const override {
    return B;
  };
};

} // end namespace gyronimo.

#endif // GYRONIMO_METRIC_CARTESIAN
