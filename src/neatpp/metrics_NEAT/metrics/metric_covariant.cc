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

// @metric_covariant.cc, this file is part of ::gyronimo::

#include <cmath>
#include <gyronimo/core/contraction.hh>
#include <gyronimo/metrics/metric_covariant.hh>
#include <iostream> 
using namespace std;

namespace gyronimo {

//! General-purpose implementation of the Jacobian.
double metric_covariant::jacobian(const IR3& r) const {
  SM3 g = (*this)(r);
  double J = g[SM3::uu]*g[SM3::vv]*g[SM3::ww] + 2.0*g[SM3::uv]*g[SM3::uw]*g[SM3::vw] -
             g[SM3::uv]*g[SM3::uv]*g[SM3::ww] - g[SM3::uu]*g[SM3::vw]*g[SM3::vw] -
             g[SM3::uw]*g[SM3::uw]*g[SM3::vv];
  double signJ = (J < 0) ? -1 : 1;
  return  signJ * std::sqrt( signJ * J);
}

//! General-purpose implementation of the Jacobian gradient.
IR3 metric_covariant::del_jacobian(const IR3& r) const {
  SM3 g = (*this)(r);
  dSM3 dg = this->del(r);
  return {
      2.0*g[SM3::vw]*(
          g[SM3::uw]*dg[dSM3::uvu] +
          g[SM3::uv]*dg[dSM3::uwu] - g[SM3::uu]*dg[dSM3::vwu]) -
      g[SM3::uv]*g[SM3::uv]*dg[dSM3::wwu] + g[SM3::vv]*(
          g[SM3::ww]*dg[dSM3::uuu] -
          2.0*g[SM3::uw]*dg[dSM3::uwu] + g[SM3::uu]*dg[dSM3::wwu]) -
      g[SM3::vw]*g[SM3::vw]*dg[dSM3::uuu] -
      2.0*g[SM3::uv]*g[SM3::ww]*dg[dSM3::uvu] - 
      g[SM3::uw]*g[SM3::uw]*dg[dSM3::vvu] +
      g[SM3::uu]*g[SM3::ww]*dg[dSM3::vvu] + 
      2.0*g[SM3::uv]*g[SM3::uw]*dg[dSM3::vwu],
      2.0*g[SM3::vw]*(
          g[SM3::uw]*dg[dSM3::uvv] +
          g[SM3::uv]*dg[dSM3::uwv] - g[SM3::uu]*dg[dSM3::vwv]) -
      g[SM3::uv]*g[SM3::uv]*dg[dSM3::wwv] + g[SM3::vv]*(
          g[SM3::ww]*dg[dSM3::uuv] -
          2.0*g[SM3::uw]*dg[dSM3::uwv] + g[SM3::uu]*dg[dSM3::wwv]) -
      g[SM3::vw]*g[SM3::vw]*dg[dSM3::uuv] -
      2.0*g[SM3::uv]*g[SM3::ww]*dg[dSM3::uvv] - 
      g[SM3::uw]*g[SM3::uw]*dg[dSM3::vvv] +
      g[SM3::uu]*g[SM3::ww]*dg[dSM3::vvv] + 
      2.0*g[SM3::uv]*g[SM3::uw]*dg[dSM3::vwv],
      2.0*g[SM3::vw]*(
          g[SM3::uw]*dg[dSM3::uvw] +
          g[SM3::uv]*dg[dSM3::uww] - g[SM3::uu]*dg[dSM3::vww]) -
      g[SM3::uv]*g[SM3::uv]*dg[dSM3::www] + g[SM3::vv]*(
          g[SM3::ww]*dg[dSM3::uuw] -
          2.0*g[SM3::uw]*dg[dSM3::uww] + g[SM3::uu]*dg[dSM3::www]) -
      g[SM3::vw]*g[SM3::vw]*dg[dSM3::uuw] -
      2.0*g[SM3::uv]*g[SM3::ww]*dg[dSM3::uvw] - 
      g[SM3::uw]*g[SM3::uw]*dg[dSM3::vvw] +
      g[SM3::uu]*g[SM3::ww]*dg[dSM3::vvw] + 
      2.0*g[SM3::uv]*g[SM3::uw]*dg[dSM3::vww]};
}

//! General-purpose product of a covariant metric and a contravariant vector.
/*!
    Returns the *covariant* product of the *covariant* metric, evaluated at a
    contravariant position, by the *contravariant* vector `B`.
*/
IR3 metric_covariant::to_covariant(const IR3& B, const IR3& r) const {
  return contraction((*this)(r), B);
}

//! General-purpose product of a contravariant metric and a covariant vector.
/*!
    Returns the *contravariant* product of the *contravariant* metric, evaluated
    at a contravariant position, with the *covariant* vector `B`. This method is
    significantly more expensive than `to_covariant` because it involves the
    covariant-metric inversion.
*/
IR3 metric_covariant::to_contravariant(const IR3& B, const IR3& r) const {
  SM3 g = (*this)(r);
  double denominator1 = g[SM3::uw]*g[SM3::uw]*g[SM3::vv] -
      2.0*g[SM3::uv]*g[SM3::uw]*g[SM3::vw] + g[SM3::uv]*g[SM3::uv]*g[SM3::ww] +
      g[SM3::uu]*( g[SM3::vw]*g[SM3::vw] - g[SM3::vv]*g[SM3::ww]);
  double denominator2 = g[SM3::uw]*g[SM3::uw]*g[SM3::vv] -
      2.0*g[SM3::uv]*g[SM3::uw]*g[SM3::vw] + g[SM3::uu]*g[SM3::vw]*g[SM3::vw] +
      g[SM3::uv]*g[SM3::uv]*g[SM3::ww] - g[SM3::uu]*g[SM3::vv]*g[SM3::ww];
  return {(
      g[SM3::vw]*g[SM3::vw]*B[IR3::u] - g[SM3::vv]*g[SM3::ww]*B[IR3::u] +
      g[SM3::uv]*g[SM3::ww]*B[IR3::v] + g[SM3::uw]*g[SM3::vv]*B[IR3::w] -
      g[SM3::vw]*(g[SM3::uw]*B[IR3::v] + g[SM3::uv]*B[IR3::w]))/denominator1,(
      g[SM3::uv]*g[SM3::ww]*B[IR3::u] + g[SM3::uw]*g[SM3::uw]*B[IR3::v] -
      g[SM3::uu]*g[SM3::ww]*B[IR3::v] + g[SM3::uu]*g[SM3::vw]*B[IR3::w] -
      g[SM3::uw]*(g[SM3::vw]*B[IR3::u] + g[SM3::uv]*B[IR3::w]))/denominator1,(
      g[SM3::uw]*g[SM3::vv]*B[IR3::u] - g[SM3::uv]*g[SM3::vw]*B[IR3::u] -
      g[SM3::uv]*g[SM3::uw]*B[IR3::v] + g[SM3::uu]*g[SM3::vw]*B[IR3::v] +
      g[SM3::uv]*g[SM3::uv]*B[IR3::w] - g[SM3::uu]*g[SM3::vv]*B[IR3::w]
      )/denominator2};
}

//! General-purpose implementation of the inverse (i.e., contravariant metric).
SM3 metric_covariant::inverse(const IR3& r) const {
  SM3 g = (*this)(r);
  double factor = 1.0/(
      g[SM3::uw]*g[SM3::uw]*g[SM3::vv] - 2.0*g[SM3::uv]*g[SM3::uw]*g[SM3::vw] +
      g[SM3::uv]*g[SM3::uv]*g[SM3::ww] + g[SM3::uu]*(
          g[SM3::vw]*g[SM3::vw] - g[SM3::vv]*g[SM3::ww]));
  return {
    factor*(g[SM3::vw]*g[SM3::vw] - g[SM3::vv]*g[SM3::ww]),
    factor*(g[SM3::uv]*g[SM3::ww] - g[SM3::uw]*g[SM3::vw]),
    factor*(g[SM3::uw]*g[SM3::vv] - g[SM3::uv]*g[SM3::vw]),
    factor*(g[SM3::uw]*g[SM3::uw] - g[SM3::uu]*g[SM3::ww]),
    factor*(g[SM3::uu]*g[SM3::vw] - g[SM3::uv]*g[SM3::uw]),
    factor*(g[SM3::uv]*g[SM3::uv] - g[SM3::uu]*g[SM3::vv])};
}

//! General-purpose implementation of the inverse derivatives.
/*!
    Implements the rule
    @f$ \partial_k g^{ij} = - g^{im} \partial_k g_{mn} g^{nj} @f$
*/
dSM3 metric_covariant::del_inverse(const IR3& r) const {
  SM3 ig = this->inverse(r);
  dSM3 dg = contraction(ig, this->del(r), ig);
  return {
    -dg[dSM3::uuu], -dg[dSM3::uuv], -dg[dSM3::uuw],
    -dg[dSM3::uvu], -dg[dSM3::uvv], -dg[dSM3::uvw],
    -dg[dSM3::uwu], -dg[dSM3::uwv], -dg[dSM3::uww],
    -dg[dSM3::vvu], -dg[dSM3::vvv], -dg[dSM3::vvw],
    -dg[dSM3::vwu], -dg[dSM3::vwv], -dg[dSM3::vww],
    -dg[dSM3::wwu], -dg[dSM3::wwv], -dg[dSM3::www]};
}

IR3 metric_covariant::transform2cylindrical(const IR3& r) const {
  return {0.0, 0.0, 0.0};
}

} // end namespace gyronimo.
