#include "metric_vmec_interp3D.hh"
#include <iostream>
#include <vector>
#include <cmath>

using namespace gyronimo;
using namespace SPLINTER;

metric_vmec_interp3D::metric_vmec_interp3D(
    const parser_vmec *p, const interpolator1d_factory *ifactory, const double *ns, const double *ntheta, const double *nzeta) 
    : parser_(p), b0_(p->B_0()), mnmax_(p->mnmax()), mnmax_nyq_(p->mnmax_nyq()),
      ns_(p->ns()), mpol_(p->mpol()), ntor_(p->ntor()), 
      signsgs_(p->signgs()), nfp_(p->nfp()),
      xm_(p->xm()), xn_(p->xn()), xm_nyq_(p->xm_nyq()), xn_nyq_(p->xn_nyq()),
      theta_modulus_factor_(2*std::numbers::pi), phi_modulus_factor_(2*std::numbers::pi/p->nfp())
{
    // Set radial grid block
    dblock_adapter s_range(p->radius());
    dblock_adapter s_half_range(p->radius_half());

    // Set spectral components 
    Rmnc_ = new interpolator1d* [xm_.size()];
    Zmns_ = new interpolator1d* [xm_.size()];
    gmnc_ = new interpolator1d* [xm_.size()];

    #pragma omp parallel for
    for(size_t i = 0; i < xm_.size(); ++i) {
        std::slice s_cut = std::slice(i, s_range.size(), xm_.size());
        std::valarray<double> rmnc_i = (p->rmnc())[s_cut];
        Rmnc_[i] = ifactory->interpolate_data(s_range, dblock_adapter(rmnc_i));
        std::valarray<double> zmnc_i = (p->zmns())[s_cut];
        Zmns_[i] = ifactory->interpolate_data(s_range, dblock_adapter(zmnc_i));

        // Note that gmnc is defined at half mesh
        std::slice s_h_cut = std::slice(i + xm_nyq_.size(), s_half_range.size(), xm_nyq_.size());
        std::valarray<double> gmnc_i = (p->gmnc())[s_h_cut];
        gmnc_[i] = ifactory->interpolate_data(s_half_range, dblock_adapter(gmnc_i));
    }

    // Create new DataTable to manage samples
    DataTable J_samples;
    DataTable del_J_samples_u, del_J_samples_v, del_J_samples_w;
    DataTable transform2cylindrical_samples_u, transform2cylindrical_samples_v, transform2cylindrical_samples_w;
    DataTable metric_vmec_samples_uu,metric_vmec_samples_uv,metric_vmec_samples_uw, metric_vmec_samples_vv, metric_vmec_samples_vw, metric_vmec_samples_ww;
    DenseVector x(3);
    IR3 del_jacobian_temp = {0, 0, 0};
    IR3 transform2cylindrical_temp = {0, 0, 0};
    SM3 metric_vmec_temp = {0, 0, 0, 0, 0, 0};
    double y;
    dblock_adapter s_radius = (parser_->radius_half());
    auto s_min = s_radius[0];
    auto s_max = s_radius[s_radius.size() - 1];
    auto ds = (s_max - s_min) / (*ns - 1);
    auto dtheta = theta_modulus_factor_ / *ntheta;
    auto dzeta = phi_modulus_factor_ / *nzeta;

    for (size_t i = 0; i < *ns; ++i) {
        for (size_t j = 0; j < *ntheta; ++j) {
            for (size_t k = 0; k < *nzeta; ++k) {
                x(0) = s_min + i * ds;
                x(1) = k * dzeta;
                x(2) = j * dtheta;
                IR3 pos = {x(0), x(1), x(2)};

                J_samples.addSample(x, jacobian_vmec(pos));

                del_jacobian_temp = del_jacobian_vmec(pos);
                del_J_samples_u.addSample(x, del_jacobian_temp[IR3::u]);
                del_J_samples_v.addSample(x, del_jacobian_temp[IR3::v]);
                del_J_samples_w.addSample(x, del_jacobian_temp[IR3::w]);

                transform2cylindrical_temp = transform2cylindrical_vmec(pos);
                transform2cylindrical_samples_u.addSample(x, transform2cylindrical_temp[IR3::u]);
                transform2cylindrical_samples_v.addSample(x, transform2cylindrical_temp[IR3::v]);
                transform2cylindrical_samples_w.addSample(x, transform2cylindrical_temp[IR3::w]);

                metric_vmec_temp = metric_vmec(pos);
                metric_vmec_samples_uu.addSample(x, metric_vmec_temp[SM3::uu]);
                metric_vmec_samples_uv.addSample(x, metric_vmec_temp[SM3::uv]);
                metric_vmec_samples_uw.addSample(x, metric_vmec_temp[SM3::uw]);
                metric_vmec_samples_vv.addSample(x, metric_vmec_temp[SM3::vv]);
                metric_vmec_samples_vw.addSample(x, metric_vmec_temp[SM3::vw]);
                metric_vmec_samples_ww.addSample(x, metric_vmec_temp[SM3::ww]);
            }
        }
    }
    jacobian_spline_       = new BSpline(BSpline::Builder(J_samples).degree(3).build());

    del_jacobian_spline_u_ = new BSpline(BSpline::Builder(del_J_samples_u).degree(3).build());
    del_jacobian_spline_v_ = new BSpline(BSpline::Builder(del_J_samples_v).degree(3).build());
    del_jacobian_spline_w_ = new BSpline(BSpline::Builder(del_J_samples_w).degree(3).build());

    transform2cylindrical_spline_u_ = new BSpline(BSpline::Builder(transform2cylindrical_samples_u).degree(3).build());
    transform2cylindrical_spline_v_ = new BSpline(BSpline::Builder(transform2cylindrical_samples_v).degree(3).build());
    transform2cylindrical_spline_w_ = new BSpline(BSpline::Builder(transform2cylindrical_samples_w).degree(3).build());

    metric_vmec_spline_uu_ = new BSpline(BSpline::Builder(metric_vmec_samples_uu).degree(3).build());
    metric_vmec_spline_uv_ = new BSpline(BSpline::Builder(metric_vmec_samples_uv).degree(3).build());
    metric_vmec_spline_uw_ = new BSpline(BSpline::Builder(metric_vmec_samples_uw).degree(3).build());
    metric_vmec_spline_vv_ = new BSpline(BSpline::Builder(metric_vmec_samples_vv).degree(3).build());
    metric_vmec_spline_vw_ = new BSpline(BSpline::Builder(metric_vmec_samples_vw).degree(3).build());
    metric_vmec_spline_ww_ = new BSpline(BSpline::Builder(metric_vmec_samples_ww).degree(3).build());
}

metric_vmec_interp3D::~metric_vmec_interp3D() {
  if(Rmnc_) delete Rmnc_;
  if(Zmns_) delete Zmns_;
  if(gmnc_) delete gmnc_;

  if(jacobian_spline_) delete jacobian_spline_;

  if(del_jacobian_spline_u_) delete del_jacobian_spline_u_;
  if(del_jacobian_spline_v_) delete del_jacobian_spline_v_;
  if(del_jacobian_spline_w_) delete del_jacobian_spline_w_;

  if(transform2cylindrical_spline_u_) delete transform2cylindrical_spline_u_;
  if(transform2cylindrical_spline_v_) delete transform2cylindrical_spline_v_;
  if(transform2cylindrical_spline_w_) delete transform2cylindrical_spline_w_;

  if(metric_vmec_spline_uu_) delete metric_vmec_spline_uu_;
  if(metric_vmec_spline_uv_) delete metric_vmec_spline_uv_;
  if(metric_vmec_spline_uw_) delete metric_vmec_spline_uw_;
  if(metric_vmec_spline_vv_) delete metric_vmec_spline_vv_;
  if(metric_vmec_spline_vw_) delete metric_vmec_spline_vw_;
  if(metric_vmec_spline_ww_) delete metric_vmec_spline_ww_;
}

double metric_vmec_interp3D::reduce_theta(double theta) const {
  return std::fmod(theta, theta_modulus_factor_);
}

double metric_vmec_interp3D::reduce_phi(double phi) const {
  return std::fmod(phi, phi_modulus_factor_);
}

SM3 metric_vmec_interp3D::metric_vmec(const IR3& position) const {
  double s = position[IR3::u], zeta = position[IR3::v], theta = position[IR3::w];
  double R = 0.0, Z = 0.0, dR_ds = 0.0, dZ_ds = 0.0, dR_dtheta = 0.0, dR_dzeta = 0.0, dZ_dtheta = 0.0, dZ_dzeta = 0.0;

  #pragma omp parallel for reduction(+: R, Z, dR_ds, dZ_ds, dR_dtheta, dR_dzeta, dZ_dtheta, dZ_dzeta)
  for (size_t i = 0; i < xm_.size(); ++i) {
    double m = xm_[i], n = xn_[i], angle = m * theta - n * zeta;
    double cosmn = std::cos(angle);
    double sinmn = std::sin(angle);
    double rmnc_i = (*Rmnc_[i])(s), zmns_i = (*Zmns_[i])(s);
    double d_rmnc_i = (*Rmnc_[i]).derivative(s), d_zmns_i = (*Zmns_[i]).derivative(s);
    R += rmnc_i * cosmn; Z += zmns_i * sinmn;
    dR_ds += d_rmnc_i * cosmn; dZ_ds += d_zmns_i * sinmn;
    double m_sin = m * sinmn, n_sin = n * sinmn;
    dR_dtheta -= m_sin * rmnc_i; dR_dzeta += n_sin * rmnc_i;
    dZ_dtheta += m_sin * zmns_i; dZ_dzeta -= n_sin * zmns_i;
  }

  return { dR_ds * dR_ds + dZ_ds * dZ_ds, dR_ds * dR_dzeta + dZ_ds * dZ_dzeta, dR_ds * dR_dtheta + dZ_ds * dZ_dtheta,
           R * R + dR_dzeta * dR_dzeta + dZ_dzeta * dZ_dzeta, dR_dtheta * dR_dzeta + dZ_dtheta * dZ_dzeta,
           dR_dtheta * dR_dtheta + dZ_dtheta * dZ_dtheta };
}

SM3 metric_vmec_interp3D::operator()(const IR3& position) const {
  DenseVector x(3);
  x(0) = position[IR3::u];
  x(1) = this->reduce_phi(position[IR3::v]);
  x(2) = this->reduce_theta(position[IR3::w]);
  SM3 metric_vmec_temp = {metric_vmec_spline_uu_->eval(x),metric_vmec_spline_uv_->eval(x),metric_vmec_spline_uw_->eval(x),
                          metric_vmec_spline_vv_->eval(x),metric_vmec_spline_vw_->eval(x),metric_vmec_spline_ww_->eval(x)};
  return metric_vmec_temp;
}

dSM3 metric_vmec_interp3D::del(const IR3& position) const {
    double s = position[IR3::u];
    double zeta = position[IR3::v];
    double theta = position[IR3::w];

    double dR_ds = 0.0, dR_dtheta = 0.0, dR_dzeta = 0.0;
    double d2R_ds2 = 0.0, d2R_dsdtheta = 0.0, d2R_dsdzeta = 0.0, d2R_dtheta2 = 0.0, d2R_dthetadzeta = 0.0, d2R_dzeta2 = 0.0;
    double dZ_ds = 0.0, dZ_dtheta = 0.0, dZ_dzeta = 0.0;
    double d2Z_ds2 = 0.0, d2Z_dsdtheta = 0.0, d2Z_dsdzeta = 0.0, d2Z_dtheta2 = 0.0, d2Z_dthetadzeta = 0.0, d2Z_dzeta2 = 0.0;

    #pragma omp parallel for reduction(+: dR_ds, dR_dtheta, dR_dzeta, d2R_ds2, d2R_dsdtheta, d2R_dsdzeta, d2R_dtheta2, d2R_dthetadzeta, d2R_dzeta2, dZ_ds, dZ_dtheta, dZ_dzeta, d2Z_ds2, d2Z_dsdtheta, d2Z_dsdzeta, d2Z_dtheta2, d2Z_dthetadzeta, d2Z_dzeta2)
    for (size_t i = 0; i < xm_.size(); i++) {
        double m = xm_[i];
        double n = xn_[i];
        double cosmn = std::cos( m*theta - n*zeta );
        double sinmn = std::sin( m*theta - n*zeta );

        double rmnc_i = (*Rmnc_[i])(s);
        double zmns_i = (*Zmns_[i])(s);
        double d_rmnc_i = (*Rmnc_[i]).derivative(s);
        double d_zmns_i = (*Zmns_[i]).derivative(s);
        double d2_rmnc_i = (*Rmnc_[i]).derivative2(s);
        double d2_zmns_i = (*Zmns_[i]).derivative2(s);

        double m_sin = m * sinmn, m_cos = m * cosmn, n_sin = n * sinmn, n_cos = n * cosmn;

        dR_ds += d_rmnc_i * cosmn;
        dR_dtheta -= m_sin * rmnc_i;
        dR_dzeta += n_sin * rmnc_i;
        d2R_ds2 += d2_rmnc_i * cosmn;
        d2R_dsdtheta -= m_sin * d_rmnc_i;
        d2R_dsdzeta += n_sin * d_rmnc_i;
        d2R_dtheta2 -= m * m_cos * rmnc_i;
        d2R_dthetadzeta += m_cos * n * rmnc_i;
        d2R_dzeta2 -= n_cos * n * rmnc_i;
        
        dZ_ds += d_zmns_i * sinmn;
        dZ_dtheta += m_cos * zmns_i;
        dZ_dzeta -= n_cos * zmns_i;
        d2Z_ds2 += d2_zmns_i * sinmn;
        d2Z_dsdtheta += m_cos * d_zmns_i;
        d2Z_dsdzeta -= n_cos * d_zmns_i;
        d2Z_dtheta2 -= m_sin * m * zmns_i;
        d2Z_dthetadzeta -= m_sin * n * zmns_i;
        d2Z_dzeta2 += n_sin * n * zmns_i;
    }

    return {
        2 * (dR_ds * d2R_ds2 + dZ_ds * d2Z_ds2),
        2 * (dR_ds * d2R_dsdzeta + dZ_ds * d2Z_dsdzeta),
        2 * (dR_ds * d2R_dsdtheta + dZ_ds * d2Z_dsdtheta),
        dR_ds * d2R_dtheta2 + dZ_ds * d2Z_dtheta2,
        dR_ds * d2R_dthetadzeta + dZ_ds * d2Z_dthetadzeta,
        dR_ds * d2R_dzeta2 + dZ_ds * d2Z_dzeta2,
        dR_dtheta * d2R_ds2 + dZ_dtheta * d2Z_ds2,
        dR_dtheta * d2R_dsdzeta + dZ_dtheta * d2Z_dsdzeta,
        dR_dtheta * d2R_dsdtheta + dZ_dtheta * d2Z_dsdtheta,
        dR_dtheta * d2R_dtheta2 + dZ_dtheta * d2Z_dtheta2,
        dR_dtheta * d2R_dthetadzeta + dZ_dtheta * d2Z_dthetadzeta,
        dR_dtheta * d2R_dzeta2 + dZ_dtheta * d2Z_dzeta2,
        dR_dzeta * d2R_ds2 + dZ_dzeta * d2Z_ds2,
        dR_dzeta * d2R_dsdzeta + dZ_dzeta * d2Z_dsdzeta,
        dR_dzeta * d2R_dsdtheta + dZ_dzeta * d2Z_dsdtheta,
        dR_dzeta * d2R_dtheta2 + dZ_dzeta * d2Z_dtheta2,
        dR_dzeta * d2R_dthetadzeta + dZ_dzeta * d2Z_dthetadzeta,
        dR_dzeta * d2R_dzeta2 + dZ_dzeta * d2Z_dzeta2
    };
}

IR3 metric_vmec_interp3D::transform2cylindrical_vmec(const IR3& position) const {
    double u = position[gyronimo::IR3::u];
    double v = position[gyronimo::IR3::v];
    double w = position[gyronimo::IR3::w];
    double R = 0.0, Z = 0.0;
  
    #pragma omp parallel for reduction(+: R, Z)
    for (size_t i = 0; i<xm_.size(); i++) {
      double m = xm_[i]; double n = xn_[i];
      R+= (*Rmnc_[i])(u) * std::cos( m*w - n*v ); 
      Z+= (*Zmns_[i])(u) * std::sin( m*w - n*v );
    }
    return  {R, v, Z};
}

IR3 metric_vmec_interp3D::transform2cylindrical(const IR3& position) const {
  DenseVector x(3);
  x(0) = position[IR3::u];
  x(1) = this->reduce_phi(position[IR3::v]);
  x(2) = this->reduce_theta(position[IR3::w]);
  IR3 transform2cylindrical_temp = {transform2cylindrical_spline_u_->eval(x),
                                    transform2cylindrical_spline_v_->eval(x),
                                    transform2cylindrical_spline_w_->eval(x)};
  return transform2cylindrical_temp;
}
 
double metric_vmec_interp3D::jacobian_vmec(const IR3& position) const {
  double s = position[IR3::u];
  double zeta = position[IR3::v];
  double theta = position[IR3::w];
  double J = 0.0;
  #pragma omp parallel for reduction(+: J)
  for (size_t i = 0; i < xm_nyq_.size(); i++) {  
    J += (*gmnc_[i])(s) * std::cos( xm_nyq_[i]*theta - xn_nyq_[i]*zeta );
  };
  return -J;
}

IR3 metric_vmec_interp3D::del_jacobian_vmec(const IR3& position) const {
    double s = position[IR3::u];
    double zeta = position[IR3::v];
    double theta = position[IR3::w];

    double dJds = 0.0;
    double dJdtheta = 0.0;
    double dJdzeta = 0.0;

    // Combine the parallel regions into a single one
    #pragma omp parallel for reduction(+:dJds, dJdtheta, dJdzeta)
    for (size_t i = 0; i < xm_nyq_.size(); i++) {
        // Use a local variable to reduce repeated calculations
        double angle = xm_nyq_[i] * theta - xn_nyq_[i] * zeta;
        double cos_angle = std::cos(angle);
        double sin_angle = std::sin(angle);

        // compute gmnc derivatives and values once
        double gmnc_der_s = (*gmnc_[i]).derivative(s);
        double gmnc_val_s = (*gmnc_[i])(s);

        // Compute and accumulate dJds, dJdtheta, dJdzeta
        dJds += gmnc_der_s * cos_angle;
        dJdtheta += -gmnc_val_s * sin_angle * xm_nyq_[i];
        dJdzeta += gmnc_val_s * sin_angle * xn_nyq_[i];
    }

    return {-dJds, -dJdzeta, -dJdtheta};
}

double metric_vmec_interp3D::jacobian(const IR3& position) const {
  DenseVector x(3);
  x(0) = position[IR3::u];
  x(1) = this->reduce_phi(position[IR3::v]);
  x(2) = this->reduce_theta(position[IR3::w]);
  return jacobian_spline_->eval(x);
}

IR3 metric_vmec_interp3D::del_jacobian(const IR3& position) const {
  DenseVector x(3);
  x(0) = position[IR3::u];
  x(1) = this->reduce_phi(position[IR3::v]);
  x(2) = this->reduce_theta(position[IR3::w]);
  IR3 del_jacobian_temp = {del_jacobian_spline_u_->eval(x),
                           del_jacobian_spline_v_->eval(x),
                           del_jacobian_spline_w_->eval(x)};
  return del_jacobian_temp;
}