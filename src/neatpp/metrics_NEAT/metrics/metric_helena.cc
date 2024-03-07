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

// @metric_helena.cc, this file is part of ::gyronimo::

#include <numbers>
#include <gyronimo/core/transpose.hh>
#include <gyronimo/metrics/metric_helena.hh>

namespace gyronimo{

metric_helena::metric_helena(
    const parser_helena *p, const interpolator2d_factory *ifactory)
    : parser_(p), R0_(p->rmag()), squaredR0_(p->rmag()*p->rmag()),
      guu_(nullptr), guv_(nullptr), gvv_(nullptr), gww_(nullptr) {
  dblock_adapter s_range(p->s()), chi_range(p->chi());
  guu_ = ifactory->interpolate_data(
      s_range, chi_range, dblock_adapter(p->covariant_g11()));
  guv_ = ifactory->interpolate_data(
      s_range, chi_range, dblock_adapter(p->covariant_g12()));
  gvv_ = ifactory->interpolate_data(
      s_range, chi_range, dblock_adapter(p->covariant_g22()));
  gww_ = ifactory->interpolate_data(
      s_range, chi_range, dblock_adapter(p->covariant_g33()));
}
metric_helena::~metric_helena() {
  if(guu_) delete guu_;
  if(guv_) delete guv_;
  if(gvv_) delete gvv_;
  if(gww_) delete gww_;
}
SM3 metric_helena::operator()(const IR3& position) const {
  double s = position[IR3::u];
  double chi = this->reduce_chi(position[IR3::v]);
  return {
      squaredR0_*(*guu_)(s, chi), squaredR0_*(*guv_)(s, chi), 0.0,
      squaredR0_*(*gvv_)(s, chi), 0.0, squaredR0_*(*gww_)(s, chi)};
}
dSM3 metric_helena::del(const IR3& position) const {
  double s = position[IR3::u];
  double chi = this->reduce_chi(position[IR3::v]);
  return {
      squaredR0_*(*guu_).partial_u(s, chi),
      squaredR0_*(*guu_).partial_v(s, chi), 0.0, // d_i g_uu
      squaredR0_*(*guv_).partial_u(s, chi),
      squaredR0_*(*guv_).partial_v(s, chi), 0.0, // d_i g_uv
      0.0, 0.0, 0.0, // d_i g_uw
      squaredR0_*(*gvv_).partial_u(s, chi),
      squaredR0_*(*gvv_).partial_v(s, chi), 0.0, //d_i g_vv
      0.0, 0.0, 0.0, // d_i g_vw
      squaredR0_*(*gww_).partial_u(s, chi),
      squaredR0_*(*gww_).partial_v(s, chi), 0.0}; // d_i g_ww
}

//! Reduces an arbitrary angle chi to the interval [0:pi].
double metric_helena::reduce_chi(double chi) const {
  chi -= 2*std::numbers::pi*std::floor(chi/(2*std::numbers::pi));
  if(parser_->is_symmetric() && chi > std::numbers::pi)
      chi = 2*std::numbers::pi - chi;
  return chi;
}

} // end namespace gyronimo
