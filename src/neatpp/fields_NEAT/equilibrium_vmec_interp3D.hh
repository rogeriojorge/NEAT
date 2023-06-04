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

// @equilibrium_vmec.hh, this file is part of ::gyronimo::

#ifndef GYRONIMO_EQUILIBRIUM_VMEC_INTERP3D
#define GYRONIMO_EQUILIBRIUM_VMEC_INTERP3D

#include <gyronimo/fields/IR3field_c1.hh>
#include "metric_vmec_interp3D.hh"
#include <gyronimo/parsers/parser_vmec.hh>
#include <gyronimo/interpolators/interpolator2d.hh>

namespace gyronimo{

//! Equilibrium magnetic field in 'VMEC' curvilinear coordinates.
/*!
    Following IR3Field rules, the magnetic field is normalised by a `m_factor`
    matching its on-axis value `B0()` in [T], which is located at `R0()` in [m].
    The coordinates are set by the `metric_vmec` object and the type of 1d
    interpolators is set by the specific `interpolator1d_factory` supplied.
    Contravariant components have dimensions of [m^{-1}]. Being an
    **equilibrium** field, `t_factor` is set to one.
    
    Only the minimal interface is implemented for the moment and further
    specialisations may enhance the object's performance.
*/
class equilibrium_vmec_interp3D : public IR3field_c1{
 public:
  typedef std::valarray<double> narray_type;
  typedef std::vector<interpolator1d> spectralarray_type;
  equilibrium_vmec_interp3D(
      const metric_vmec_interp3D *g, const interpolator1d_factory *ifactory);
  virtual ~equilibrium_vmec_interp3D() override;

  virtual IR3 contravariant(const IR3& position, double time) const override;
  virtual dIR3 del_contravariant(
      const IR3& position, double time) const override;
  virtual IR3 partial_t_contravariant(
      const IR3& position, double time) const override {return {0.0,0.0,0.0};};
  double magnitude_vmec(const IR3& position, double time) const;

  double R_0() const {return metric_->parser()->R_0();};
  double B_0() const {return metric_->parser()->B_0();};
  const metric_vmec_interp3D* metric() const {return metric_;};
 private:
  const metric_vmec_interp3D *metric_;
  narray_type xm_nyq_, xn_nyq_; 
  interpolator1d **bmnc_;
  interpolator1d **bsupumnc_;
  interpolator1d **bsupvmnc_;
};

}// end namespace gyronimo.

#endif // GYRONIMO_EQUILIBRIUM_VMEC
