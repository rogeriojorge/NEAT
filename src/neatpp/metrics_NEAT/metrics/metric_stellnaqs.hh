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

// @metric_stellnaqs.hh, this file is part of ::gyronimo::

#ifndef GYRONIMO_METRIC_STELLNAQS
#define GYRONIMO_METRIC_STELLNAQS

#include <gyronimo/metrics/metric_covariant.hh>
#include <gyronimo/interpolators/interpolator1d.hh>

namespace gyronimo{

//! Quasi-symmetric stellarator Boozer coordinates via near-axis expansion.
/*!
    Implements the coordinate set @f$\{r, \vartheta, \varphi\}@f$ as defined in
    M.~Landreman and W.~Sengupta, J.~Plasma.~Phys. **85**, 815850601 (2019).
    Metric coefficients are obtained taking derivatives of the transformation in
    equation 2.13, which is expanded around a magnetic axis defined (as any line
    in @f$\mathbb{R}^3@f$) by some curvature, torsion, and arclength (actually
    the derivative @f$dl/d\varphi@f$) functions of the toroidal angle
    @f$\varphi@f$. These functions are solutions of the near-axis equation set
    and can be obtained from specialised tools like
    [pyQSC](https://github.com/landreman/pyQSC). The constants `eta_bar` and
    `field_periods` are as defined in the cited reference.
*/
class metric_stellnaqs : public metric_covariant {
 public:
  metric_stellnaqs(
      int field_periods, double eta_bar,
      const dblock& phi_grid, const dblock& sigma,
      const dblock& dldphi, const dblock& torsion, const dblock& curvature,
      const interpolator1d_factory* ifactory);
  virtual ~metric_stellnaqs() override;

  virtual SM3 operator()(const IR3& position) const override;
  virtual dSM3 del(const IR3& position) const override;

  const interpolator1d* curvature() const {return curvature_;};
  const interpolator1d* torsion() const {return torsion_;};
  const interpolator1d* dldphi() const {return dldphi_;};
  const interpolator1d* sigma() const {return sigma_;};
  int field_periods() const {return field_periods_;};
  double reduce_phi(double phi) const;

 private:
  const int field_periods_;
  const double eta_bar_, phi_modulus_factor_;
  interpolator1d *sigma_, *curvature_, *torsion_, *dldphi_;
};

} // end namespace gyronimo

#endif // GYRONIMO_METRIC_STELLNAQS
