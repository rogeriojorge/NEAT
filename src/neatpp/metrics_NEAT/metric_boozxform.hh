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

// @metric_boozxform.hh, this file is part of ::gyronimo::

#ifndef GYRONIMO_METRIC_BOOZXFORM
#define GYRONIMO_METRIC_BOOZXFORM

#include <gyronimo/parsers/parser_boozxform.hh>
#include <gyronimo/metrics/metric_covariant.hh>
#include <gyronimo/interpolators/interpolator1d.hh>

namespace gyronimo{

//! Covariant metric in `Boozxform` curvilinear coordinates.
class metric_boozxform : public metric_covariant {
 public:
  typedef std::valarray<double> narray_type;
  typedef std::vector<interpolator1d> spectralarray_type;

  metric_boozxform(
    const parser_boozxform *parser, const interpolator1d_factory *ifactory);
  virtual ~metric_boozxform() override;
  virtual SM3 operator()(const IR3& position) const override;
  virtual dSM3 del(const IR3& position) const override;
  virtual IR3 transform2cylindrical(const IR3& position) const override;
  virtual double jacobian(const IR3& position) const override;
  virtual IR3 del_jacobian(const IR3& position) const override;
  const parser_boozxform* parser() const {return parser_;};
  // const double signgs() const {return signsgs_;};

 private:
  const parser_boozxform* parser_;
  double b0_;
  int mnboz_b_, ns_b_, mboz_b_, nboz_b_, nfp_b_; 
  narray_type ixm_b_, ixn_b_; 
  interpolator1d **Rmnc_b_;
  interpolator1d **Zmns_b_;
  interpolator1d **gmnc_b_;
};

} // end namespace gyronimo

#endif // GYRONIMO_METRIC_BOOZXFORM
