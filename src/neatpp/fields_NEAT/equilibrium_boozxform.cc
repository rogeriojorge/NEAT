// ::gyronimo:: - gyromotion for the people, by the people -
// An object-oriented library for gyromotion applications in plasma physics.
// Copyright (C) 2021 Paulo Rodrigues, Rogerio Jorge.

// @equilibrium_boozxform.cc

#include <gyronimo/core/dblock.hh>
#include <equilibrium_boozxform.hh>
#include <iostream>

namespace gyronimo{

equilibrium_boozxform::equilibrium_boozxform(
    const metric_boozxform *g, const interpolator1d_factory *ifactory)
    : IR3field_c1(abs(g->parser()->B_0()), 1.0, g), metric_(g), iota_b_(nullptr) {

  const parser_boozxform *p = metric_->parser();
  dblock_adapter s_range(p->radius());
  iota_b_ = ifactory->interpolate_data(s_range, dblock_adapter((p->iota_b())));

}

equilibrium_boozxform::~equilibrium_boozxform() {
  if (iota_b_) delete iota_b_;
}

IR3 equilibrium_boozxform::contravariant(const IR3& position, double time) const {
  // double s = position[IR3::u];
  // using namespace std;
  // // double iota_0=(((metric())->parser())->iota_b())[1];
  // return {0.0,  1.0/((this->m_factor())*(this->m_factor())), (*iota_b_)(s)/((this->m_factor())*(this->m_factor()))};
  double s    = position[IR3::u];
  double jac   = metric_->jacobian(position);
  #include <iostream>
  using namespace std;

  // cout << (*iota_b_)(s) << endl;
//   double Bu    = 0;
//   double Bv    = r*metric_->Bref()*(*iota_b_)(s)/jac;
  double Bv = (1/sqrt(jac));
  return {0, Bv/ ((this->m_factor())), (*iota_b_)(s)*Bv/ (this->m_factor())};
}

dIR3 equilibrium_boozxform::del_contravariant(
    const IR3& position, double time) const {

  double s = position[IR3::u];
  double jac = metric_->jacobian(position);
  double d_u_jac = metric_->del_jacobian(position)[IR3::u];
  double d_v_jac = metric_->del_jacobian(position)[IR3::v];
  double d_w_jac = metric_->del_jacobian(position)[IR3::w];
  #include <iostream>
  using namespace std;
  // cout << jac << d_u_jac << d_v_jac << d_w_jac << (this->m_factor()) << endl;
  double Bv = 1/jac;
  double d_u_Bv = -d_u_jac/(jac*jac);
  double d_v_Bv = -d_v_jac/(jac*jac);
  double d_w_Bv = -d_w_jac/(jac*jac);

  return {
      0, 0, 0,
      d_u_Bv/ this->m_factor(), 
      d_v_Bv/ this->m_factor(), 
      d_w_Bv/ this->m_factor(),
      (*iota_b_).derivative(s)*Bv/ this->m_factor() + (*iota_b_)(s)*d_u_Bv/ this->m_factor(), 
      (*iota_b_)(s)*d_v_Bv/ this->m_factor(), 
      (*iota_b_)(s)*d_w_Bv/ this->m_factor()};
}

// IR3 equilibrium_boozxform::covariant(const IR3& position, double time) const {
//   double s = position[IR3::u];
//   double zeta = position[IR3::v];
//   double theta = position[IR3::w];
//   double B_theta = 0.0, B_zeta = 0.0, B_s = 0.0;
//   #pragma omp parallel for reduction(+: B_zeta, B_theta, B_s)
//   for (size_t i = 0; i < xm_nyq_.size(); i++) {  
//     double m = xm_nyq_[i]; double n = xn_nyq_[i];
//     double cosmn = std::cos( m*theta - n*zeta );
//     double sinmn = std::sin( m*theta - n*zeta );
//     B_zeta += (*bsubvmnc_[i])(s) * cosmn;
//     B_theta += (*bsubumnc_[i])(s) * cosmn;
//     B_s += (*bsubsmns_[i])(s) * sinmn;
//   };
//   return {B_s,  B_zeta, B_theta};
// }
// dIR3 equilibrium_boozxform::del_covariant(
//     const IR3& position, double time) const {
//   double s = position[IR3::u];
//   double zeta = position[IR3::v];
//   double theta = position[IR3::w];
//   double B_theta = 0.0, B_zeta = 0.0, B_s = 0.0;
//   double dB_theta_ds = 0.0, dB_theta_dtheta = 0.0, dB_theta_dzeta = 0.0;
//   double dB_zeta_ds = 0.0, dB_zeta_dtheta = 0.0, dB_zeta_dzeta = 0.0;
//   double dB_s_ds = 0.0, dB_s_dtheta = 0.0, dB_s_dzeta = 0.0;
//   #pragma omp parallel for reduction(+: B_zeta, B_theta, B_s, dB_theta_ds, dB_theta_dtheta, dB_theta_dzeta, dB_zeta_ds, dB_zeta_dtheta, dB_zeta_dzeta, dB_s_ds, dB_s_dtheta, dB_s_dzeta)
//   for (size_t i = 0; i < xm_nyq_.size(); i++) {  
//     double m = xm_nyq_[i]; double n = xn_nyq_[i];
//     double cosmn = std::cos( m*theta - n*zeta );
//     double sinmn = std::sin( m*theta - n*zeta );
//     double bsubumnc_i = (*bsubumnc_[i])(s);
//     double bsubvmnc_i = (*bsubvmnc_[i])(s);
//     double bsubsmns_i = (*bsubsmns_[i])(s);
//     B_theta += bsubumnc_i * cosmn; 
//     B_zeta += bsubvmnc_i * cosmn;
//     B_s += bsubsmns_i * sinmn;
//     dB_theta_ds += (*bsubumnc_[i]).derivative(s) * cosmn;
//     dB_theta_dtheta -= m * bsubumnc_i * sinmn;
//     dB_theta_dzeta += n * bsubumnc_i * sinmn;
//     dB_zeta_ds += (*bsubvmnc_[i]).derivative(s) * cosmn;
//     dB_zeta_dtheta -= m * bsubvmnc_i * sinmn;
//     dB_zeta_dzeta += n * bsubvmnc_i * sinmn;
//     dB_s_ds += (*bsubsmns_[i]).derivative(s) * sinmn;
//     dB_s_dtheta += m * bsubsmns_i * cosmn;
//     dB_s_dzeta -= n * bsubsmns_i * cosmn;
//   };
//   return {
//       dB_s_ds, dB_s_dtheta, dB_s_dzeta, 
// 	    dB_zeta_ds, dB_zeta_dzeta, dB_zeta_dtheta,
//       dB_theta_ds, dB_theta_dzeta, dB_theta_dtheta
//   };
// }
//@todo we can actually override the methods to calculate the covariant components of the field
//@todo move this to magnitude after the half radius issue is sorted out -> perguntar ao Jorge se está resolvido
// double equilibrium_boozxform::magnitude(
//     const IR3& position, double time) const {
//   double s = position[IR3::u];
//   double zeta = position[IR3::v];
//   double theta = position[IR3::w];
//   double Bnorm = 0.0;
//   #pragma omp parallel for reduction(+: Bnorm)
//   for (size_t i = 0; i < xm_nyq_.size(); i++) {  
//     Bnorm += (*bmnc_[i])(s) * std::cos( xm_nyq_[i] * theta - xn_nyq_[i] *zeta );
//   };
//   return Bnorm;
// }
// IR3 equilibrium_boozxform::del_magnitude(
//     const IR3& position, double time) const {
//   double s = position[IR3::u];
//   double zeta = position[IR3::v];
//   double theta = position[IR3::w];
//   double B_ds = 0.0, B_dzeta = 0.0, B_dtheta = 0.0;
//   #pragma omp parallel for reduction(+: B_ds, B_dzeta, B_dtheta)
//   for (size_t i = 0; i < xm_nyq_.size(); i++) {  
//     B_ds += (*bmnc_[i]).derivative(s) * std::cos( xm_nyq_[i]*theta - xn_nyq_[i]*zeta );
//     B_dzeta += xn_nyq_[i] * (*bmnc_[i])(s) * std::sin( xm_nyq_[i]*theta - xn_nyq_[i]*zeta );
//     B_dtheta -= xm_nyq_[i] * (*bmnc_[i])(s) * std::sin( xm_nyq_[i]*theta - xn_nyq_[i]*zeta );
//   };
//   return {B_ds, B_dzeta, B_dtheta};
// } 

}// end namespace gyronimo.
