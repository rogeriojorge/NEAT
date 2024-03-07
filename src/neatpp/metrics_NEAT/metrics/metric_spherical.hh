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

// @metric_spherical.hh, this file is part of ::gyronimo::

#ifndef GYRONIMO_METRIC_SPHERICAL
#define GYRONIMO_METRIC_SPHERICAL

#include <gyronimo/metrics/metric_covariant.hh>

namespace gyronimo {

//! Covariant metric for spherical coordinates.
/*!
    The three contravariant coordinates are the distance to the origin
    normalized to `radius_norm` (`u`, with `radius_norm` in SI), the polar angle
    measured from the z-axis (co-latitude `v`, in rads), and the azimuthal angle
    (`w`, also in rads) measured clockwise when looking from the origin along
    the z-axis. Some inherited methods are overriden for efficiency.
*/
class metric_spherical : public metric_covariant {
 public:
  metric_spherical(double radius_norm);
  virtual ~metric_spherical() override {};

  virtual SM3 operator()(const IR3& r) const override ;
  virtual dSM3 del(const IR3& r) const override;

  virtual double jacobian(const IR3& r) const override;
  virtual IR3 del_jacobian(const IR3& r) const override;
  virtual IR3 to_covariant(const IR3& B, const IR3& r) const override ;
  virtual IR3 to_contravariant(const IR3& B, const IR3& r) const override;
 private:
  const double radius_norm_;
  const double radius_norm_squared_, radius_norm_cube_;
};

} // end namespace gyronimo.

#endif // GYRONIMO_METRIC_SPHERICAL
