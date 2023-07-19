#include <gyronimo/core/dblock.hh>
#include <equilibrium_boozxform.hh>
#include <numbers>
#include <iostream>
using namespace std;

namespace gyronimo{

equilibrium_boozxform::equilibrium_boozxform(
    const metric_boozxform *g, const interpolator1d_factory *ifactory)
    : IR3field_c1(abs(g->parser()->B_0()), 1.0, g), metric_(g),
    ixm_b_(g->parser()->ixm_b()), ixn_b_(g->parser()->ixn_b()),
    iota_b_(nullptr), G_(nullptr), I_(nullptr), psi_boundary_(metric_->psi_boundary()) {

  const parser_boozxform *p = metric_->parser();
  dblock_adapter s_range(p->radius());
  bmnc_b_ = new interpolator1d* [ixm_b_.size()];
  double m_factor = this->m_factor();
  std::valarray<double> bmnc_b = (p->bmnc_b())/m_factor;
  // #pragma omp parallel
  for(size_t i=0; i<ixm_b_.size(); i++) {
    std::slice s_cut (i, s_range.size(), ixm_b_.size());
    std::valarray<double> bmnc_i = bmnc_b[s_cut]; // note the m_factor above
    bmnc_b_[i] = ifactory->interpolate_data( s_range, dblock_adapter(bmnc_i));
  };
  double sign_iota_change = -1; // note the minus sign for iota_b
  std::valarray<double> iota_i = sign_iota_change*(p->iota_b());
  iota_b_ = ifactory->interpolate_data(s_range, dblock_adapter(iota_i));
  std::valarray<double> bvco_i = (p->bvco_b())/m_factor; // note the m_factor
  G_ = ifactory->interpolate_data(s_range, dblock_adapter(bvco_i));
  std::valarray<double> buco_i = (p->buco_b())/m_factor; // note the m_factor
  I_ = ifactory->interpolate_data(s_range, dblock_adapter(buco_i));
  
}

equilibrium_boozxform::~equilibrium_boozxform() {
  if (bmnc_b_) delete bmnc_b_;
  if (iota_b_) delete iota_b_;
  if (G_) delete G_;
  if (I_) delete I_;
}

IR3 equilibrium_boozxform::contravariant(const IR3& position, double time) const {
  double s    = position[IR3::u];
  double jac   = metric_->jacobian(position);
  double Bv = psi_boundary_/jac/this->m_factor();
  return {0.0, (*iota_b_)(s)*Bv, Bv};
}

dIR3 equilibrium_boozxform::del_contravariant(
    const IR3& position, double time) const {

  double s = position[IR3::u];
  double jac = metric_->jacobian(position);
  IR3 del_jac = metric_->del_jacobian(position);
  double d_u_jac = del_jac[IR3::u];
  double d_v_jac = del_jac[IR3::v];
  double d_w_jac = del_jac[IR3::w];

  double Bv = psi_boundary_/jac/this->m_factor();
  double d_u_Bv = -psi_boundary_*d_u_jac/(jac*jac)/this->m_factor();
  double d_v_Bv = -psi_boundary_*d_v_jac/(jac*jac)/this->m_factor();
  double d_w_Bv = -psi_boundary_*d_w_jac/(jac*jac)/this->m_factor();

  return {
      0.0, 0.0, 0.0,
      (*iota_b_).derivative(s)*Bv + (*iota_b_)(s)*d_u_Bv, 
      (*iota_b_)(s)*d_v_Bv, 
      (*iota_b_)(s)*d_w_Bv,
      d_u_Bv, 
      d_v_Bv,
      d_w_Bv};
}

IR3 equilibrium_boozxform::covariant(const IR3& position, double time) const {
  double s = position[IR3::u];
  return {0, (*I_)(s),  (*G_)(s)};
}
dIR3 equilibrium_boozxform::del_covariant(
    const IR3& position, double time) const {
  double s = position[IR3::u];
  return {
      0.0, 0.0, 0.0, 
      (*I_).derivative(s), 
      0.0, 0.0,
	    (*G_).derivative(s), 
      0.0, 0.0
  };
}
//@todo we can actually override the methods to calculate the covariant components of the field
//@todo move this to magnitude after the half radius issue is sorted out -> perguntar ao Jorge se está resolvido
double equilibrium_boozxform::magnitude(
    const IR3& position, double time) const {
  double s = position[IR3::u];
  double theta = numbers::pi-position[IR3::v];
  double zeta = position[IR3::w];
  double Bnorm = 0.0;
  // #pragma omp parallel for reduction(+: Bnorm)
  for (size_t i = 0; i < ixm_b_.size(); i++) {  
    Bnorm += (*bmnc_b_[i])(s) * cos( ixm_b_[i] * theta - ixn_b_[i] *zeta );
  };
  return Bnorm;
}
IR3 equilibrium_boozxform::del_magnitude(
    const IR3& position, double time) const {
  double s = position[IR3::u];
  double theta = numbers::pi-position[IR3::v];
  double zeta = position[IR3::w];
  double B_ds = 0.0, B_dzeta = 0.0, B_dtheta = 0.0, sintheta = 0.0, bmnc = 0.0;
  // #pragma omp parallel for reduction(+: B_ds, B_dzeta, B_dtheta)
  for (size_t i = 0; i < ixm_b_.size(); i++) {  
    B_ds     += (*bmnc_b_[i]).derivative(s)  * cos( ixm_b_[i]*theta - ixn_b_[i]*zeta );
    sintheta = sin( ixm_b_[i]*theta - ixn_b_[i]*zeta );
    bmnc = (*bmnc_b_[i])(s);
    B_dtheta += ixm_b_[i] * bmnc * sintheta;
    B_dzeta  += ixn_b_[i] * bmnc * sintheta;
  };
  return {B_ds, B_dtheta, B_dzeta};
} 

}// end namespace gyronimo.
