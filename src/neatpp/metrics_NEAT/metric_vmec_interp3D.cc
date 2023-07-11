#include "metric_vmec_interp3D.hh"

using namespace gyronimo;
using namespace SPLINTER;

metric_vmec_interp3D::metric_vmec_interp3D(
    const parser_vmec *p, const interpolator1d_factory *ifactory,
    double ns_interp_input, double ntheta_interp_input, double nzeta_interp_input) 
    : parser_(p), b0_(p->B_0()), mnmax_(p->mnmax()), mnmax_nyq_(p->mnmax_nyq()),
      ns_(p->ns()), mpol_(p->mpol()), ntor_(p->ntor()), 
      signsgs_(p->signgs()), nfp_(p->nfp()),
      xm_(p->xm()), xn_(p->xn()), xm_nyq_(p->xm_nyq()), xn_nyq_(p->xn_nyq()),
      Rmnc_(nullptr), Zmns_(nullptr), gmnc_(nullptr),
      theta_modulus_factor_(2*std::numbers::pi), phi_modulus_factor_(2*std::numbers::pi/p->nfp()),
      ns_interp_(ns_interp_input), ntheta_interp_(ntheta_interp_input), nzeta_interp_(nzeta_interp_input)
      {
    // set radial grid block
    dblock_adapter s_range(p->radius());
    dblock_adapter s_half_range(p->radius_half());
    // set spectral components 
    Rmnc_ = new interpolator1d* [xm_.size()];
    Zmns_ = new interpolator1d* [xm_.size()];
    gmnc_ = new interpolator1d* [xm_.size()];
//@todo NEED TO FIX AXIS AND EDGE! TBI! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // #pragma omp parallel for
    for(size_t i=0; i<xm_.size(); i++) {
      std::slice s_cut (i, s_range.size(), xm_.size());
      std::valarray<double> rmnc_i = (p->rmnc())[s_cut];
      Rmnc_[i] = ifactory->interpolate_data( s_range, dblock_adapter(rmnc_i));
      std::valarray<double> zmnc_i = (p->zmns())[s_cut];
      Zmns_[i] = ifactory->interpolate_data( s_range, dblock_adapter(zmnc_i));
      // note that gmnc is defined at half mesh
      std::slice s_h_cut (i+xm_nyq_.size(), s_half_range.size(), xm_nyq_.size());
      std::valarray<double> gmnc_i = (p->gmnc())[s_h_cut];
      gmnc_[i] = ifactory->interpolate_data( s_half_range, dblock_adapter(gmnc_i));
    };

    DataTable transform2cylindrical_samples_u, transform2cylindrical_samples_v, transform2cylindrical_samples_w;
    DataTable metric_vmec_samples_uu,metric_vmec_samples_uv,metric_vmec_samples_uw, metric_vmec_samples_vv, metric_vmec_samples_vw, metric_vmec_samples_ww;
    IR3 transform2cylindrical_temp = {0, 0, 0};
    SM3 metric_vmec_temp = {0, 0, 0, 0, 0, 0};
    dblock_adapter s_radius = (parser_->radius_half());
    auto s_min = s_radius[0];
    auto s_max = s_radius[s_radius.size() - 1];
    auto ds = (s_max - s_min) / (ns_interp_ - 1);
    auto dtheta = theta_modulus_factor_ / ntheta_interp_;
    auto dzeta = phi_modulus_factor_ / nzeta_interp_;
    DenseVector x(3);
    IR3 pos = {0,0,0};

    for (size_t i = 0; i < ns_interp_; ++i) {
        for (size_t j = 0; j <= ntheta_interp_; ++j) {
            for (size_t k = 0; k <= nzeta_interp_; ++k) {
                x(0) = s_min + i * ds;
                x(1) = k * dzeta;
                x(2) = j * dtheta;
                pos = {x(0), x(1), x(2)};
                
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
    transform2cylindrical_spline_u_ = new BSpline(BSpline::Builder(transform2cylindrical_samples_u).degree(1).build());
    transform2cylindrical_spline_v_ = new BSpline(BSpline::Builder(transform2cylindrical_samples_v).degree(1).build());
    transform2cylindrical_spline_w_ = new BSpline(BSpline::Builder(transform2cylindrical_samples_w).degree(1).build());

    metric_vmec_spline_uu_ = new BSpline(BSpline::Builder(metric_vmec_samples_uu).degree(1).build());
    metric_vmec_spline_uv_ = new BSpline(BSpline::Builder(metric_vmec_samples_uv).degree(1).build());
    metric_vmec_spline_uw_ = new BSpline(BSpline::Builder(metric_vmec_samples_uw).degree(1).build());
    metric_vmec_spline_vv_ = new BSpline(BSpline::Builder(metric_vmec_samples_vv).degree(1).build());
    metric_vmec_spline_vw_ = new BSpline(BSpline::Builder(metric_vmec_samples_vw).degree(1).build());
    metric_vmec_spline_ww_ = new BSpline(BSpline::Builder(metric_vmec_samples_ww).degree(1).build());

    // // Boundaries of the interval [-pi, pi]
    // constexpr double b = 3.14159265358979323846, a = -b;

    // // Discretize the set [-pi, pi] X [-pi, pi] using 15 evenly-spaced
    // // points along the x axis and 15 evenly-spaced points along the y axis
    // // and evaluate sin(x)cos(y) at each of those points
    // constexpr int nxd = 15, nyd = 15, nd[] = { nxd, nyd };
    // double xd[nxd];
    // for(int i = 0; i < nxd; ++i) {
    //   xd[i] = a + (b - a) / (nxd - 1) * i;
    // }
    // double yd[nyd];
    // for(int j = 0; j < nyd; ++j) {
    //   yd[j] = a + (b - a) / (nyd - 1) * j;
    // }
    // double zd[nxd * nyd];
    // for(int i = 0; i < nxd; ++i) {
    //   for(int j = 0; j < nyd; ++j) {
    //     const int n = j + i * nyd;
    //     zd[n] = sin(xd[i]) * cos(yd[j]);
    //   }
    // }

    // // Subdivide the set [-pi, pi] X [-pi, pi] using 100 evenly-spaced
    // // points along the x axis and 100 evenly-spaced points along the y axis
    // // (these are the points at which we interpolate)
    // constexpr int m = 100, ni = m * m;
    // double xi[ni];
    // double yi[ni];
    // for(int i = 0; i < m; ++i) {
    //   for(int j = 0; j < m; ++j) {
    //     const int n = j + i * m;
    //     xi[n] = a + (b - a) / (m - 1) * i;
    //     yi[n] = a + (b - a) / (m - 1) * j;
    //   }
    // }

    // // Perform the interpolation
    // double zi[ni]; // Result is stored in this buffer
    // interp(
    //   nd, ni,        // Number of points
    //   zd, zi,        // Output axis (z)
    //   xd, xi, yd, yi // Input axes (x and y)
    // );

    // // Print the interpolated values
    // cout << scientific << setprecision(8) << showpos;
    // for(int n = 0; n < ni; ++n) {
    //   cout << xi[n] << "\t" << yi[n] << "\t" << zi[n] << endl;
    // }

}

metric_vmec_interp3D::~metric_vmec_interp3D() {
  if(Rmnc_) delete Rmnc_;
  if(Zmns_) delete Zmns_;
  if(gmnc_) delete gmnc_;

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
  theta = std::fmod(theta, theta_modulus_factor_);
  return (theta < 0 ? theta + theta_modulus_factor_ : theta);
}

double metric_vmec_interp3D::reduce_phi(double phi) const {
  phi = std::fmod(phi, phi_modulus_factor_);
  return (phi < 0 ? phi + phi_modulus_factor_ : phi);
}

IR3 metric_vmec_interp3D::transform2cylindrical_vmec(const IR3& position) const {
    double u = position[gyronimo::IR3::u];
    double v = position[gyronimo::IR3::v];
    double w = position[gyronimo::IR3::w];
    double R = 0.0, Z = 0.0;
  
    // #pragma omp parallel for reduction(+: R, Z)
    for (size_t i = 0; i<xm_.size(); i++) {
      double m = xm_[i]; double n = xn_[i];
      R+= (*Rmnc_[i])(u) * std::cos( m*w - n*v ); 
      Z+= (*Zmns_[i])(u) * std::sin( m*w - n*v );
    }
    return  {R, v, Z};
}

SM3 metric_vmec_interp3D::metric_vmec(const IR3& position) const {
  double s = position[IR3::u];
  double zeta = position[IR3::v];
  double theta = position[IR3::w];
  double R = 0.0, dR_ds = 0.0, dR_dtheta = 0.0, dR_dzeta = 0.0;
  double Z = 0.0, dZ_ds = 0.0, dZ_dtheta = 0.0, dZ_dzeta = 0.0;

//    #pragma omp parallel for reduction(+: R, Z, dR_ds, dR_dtheta, dR_dzeta, dZ_ds, dZ_dtheta, dZ_dzeta)
  for (size_t i = 0; i<xm_.size(); i++) {  
    double m = xm_[i]; double n = xn_[i];
    double cosmn = std::cos( m*theta - n*zeta );
    double sinmn = std::sin( m*theta - n*zeta );
    double rmnc_i = (*Rmnc_[i])(s); 
    double zmns_i = (*Zmns_[i])(s);
    // assuming for now that vmec equilibrium has stellarator symmetry.
    R += rmnc_i * cosmn; 
    Z += zmns_i * sinmn;
    dR_ds += (*Rmnc_[i]).derivative(s) * cosmn; 
    dR_dtheta -= m * rmnc_i * sinmn; 
    dR_dzeta += n * rmnc_i * sinmn;
    dZ_ds += (*Zmns_[i]).derivative(s) * sinmn; 
    dZ_dtheta += m * zmns_i * cosmn; 
    dZ_dzeta -= n * zmns_i * cosmn; 
  };
  return {
    dR_ds * dR_ds + dZ_ds * dZ_ds,                      // g_uu
    dR_ds * dR_dzeta + dZ_ds * dZ_dzeta,                // g_uw
    dR_ds * dR_dtheta + dZ_ds * dZ_dtheta,              // g_uv
    R * R + dR_dzeta * dR_dzeta + dZ_dzeta * dZ_dzeta,  // g_vv
    dR_dtheta * dR_dzeta + dZ_dtheta * dZ_dzeta,        // g_vw
    dR_dtheta * dR_dtheta + dZ_dtheta * dZ_dtheta       // g_ww
  };
}

dSM3 metric_vmec_interp3D::del_metric_vmec(const IR3& position) const {
  double s = position[IR3::u];
  double zeta = position[IR3::v];
  double theta = position[IR3::w];
  double R = 0.0, Z = 0.0;
  double dR_ds = 0.0,        dR_dtheta = 0.0,       dR_dzeta = 0.0;
  double d2R_ds2 = 0.0,      d2R_dsdtheta = 0.0,    d2R_dsdzeta = 0.0;
  double d2R_dthetads = 0.0, d2R_dtheta2 = 0.0,     d2R_dthetadzeta = 0.0; 
  double d2R_dzetads = 0.0,  d2R_dzetadtheta = 0.0, d2R_dzeta2 = 0.0;
  double dZ_ds = 0.0,        dZ_dtheta = 0.0,       dZ_dzeta = 0.0;
  double d2Z_ds2 = 0.0,      d2Z_dsdtheta = 0.0,    d2Z_dsdzeta = 0.0;
  double d2Z_dthetads = 0.0, d2Z_dtheta2 = 0.0,     d2Z_dthetadzeta = 0.0; 
  double d2Z_dzetads = 0.0,  d2Z_dzetadtheta = 0.0, d2Z_dzeta2 = 0.0;

//   #pragma omp parallel for reduction(+: R, dR_ds, dR_dtheta, dR_dzeta, d2R_ds2, d2R_dsdtheta, d2R_dsdzeta, d2R_dtheta2, d2R_dthetadzeta, d2R_dzeta2, Z, dZ_ds ,dZ_dtheta, dZ_dzeta, d2Z_ds2, d2Z_dsdtheta, d2Z_dsdzeta, d2Z_dtheta2, d2Z_dthetadzeta, d2Z_dzeta2)
  for (size_t i = 0; i<xm_.size(); i++) {  
    double m = xm_[i]; double n = xn_[i];
    double cosmn = std::cos( m*theta - n*zeta );
    double sinmn = std::sin( m*theta - n*zeta );
    double rmnc_i = (*Rmnc_[i])(s); 
    double zmns_i = (*Zmns_[i])(s);
    double d_rmnc_i = (*Rmnc_[i]).derivative(s); 
    double d_zmns_i = (*Zmns_[i]).derivative(s); 
    double d2_rmnc_i = (*Rmnc_[i]).derivative2(s);
    double d2_zmns_i = (*Zmns_[i]).derivative2(s);
    // assuming for now that vmec equilibrium has stellarator symmetry.
    R += rmnc_i * cosmn; Z += zmns_i * sinmn;
    dR_ds += d_rmnc_i * cosmn; 
    dR_dtheta -= m * rmnc_i * sinmn; 
    dR_dzeta += n * rmnc_i * sinmn;
    d2R_ds2 += d2_rmnc_i * cosmn; 
    d2R_dsdtheta -= m * d_rmnc_i * sinmn;
    d2R_dsdzeta += n * d_rmnc_i * sinmn;
    d2R_dtheta2 -= m * m * rmnc_i * cosmn;
    d2R_dthetadzeta += m * n * rmnc_i * cosmn;
    d2R_dzeta2 -= n * n * rmnc_i * cosmn;
    dZ_ds += d_zmns_i * sinmn; 
    dZ_dtheta += m * zmns_i * cosmn; 
    dZ_dzeta -= n * zmns_i * cosmn; 
    d2Z_ds2 += d2_zmns_i * sinmn;
    d2Z_dsdtheta += m * d_zmns_i * cosmn;
    d2Z_dsdzeta -= n * d_zmns_i * cosmn;
    d2Z_dtheta2 -= m * m * zmns_i * sinmn;
    d2Z_dthetadzeta += m * n * zmns_i * sinmn;
    d2Z_dzeta2 -= n * n * zmns_i * sinmn;
}
//@todo still need to test this carefully. Find a way to test d_g!
  return {
      2 * (dR_ds * d2R_ds2      + dZ_ds * d2Z_ds2), 
      2 * (dR_ds * d2R_dsdzeta  + dZ_ds * d2Z_dsdzeta), // d_i g_uu
      2 * (dR_ds * d2R_dsdtheta + dZ_ds * d2Z_dsdtheta), 
      dR_ds * d2R_dsdzeta       + dR_dzeta * d2R_ds2      + dZ_ds * d2Z_dsdzeta      + dZ_dzeta * d2Z_ds2,
      dR_ds * d2R_dzeta2        + dR_dzeta * d2R_dsdzeta  + dZ_ds * d2Z_dzeta2       + dZ_dzeta * d2Z_dsdzeta,// d_i g_uv
      dR_ds * d2R_dthetadzeta   + dR_dzeta * d2R_dsdtheta  + dZ_ds * d2Z_dthetadzeta  + dZ_dzeta * d2Z_dsdtheta, 
      dR_ds * d2R_dsdtheta      + dR_dtheta * d2R_ds2      + dZ_ds * d2Z_dsdtheta     + dZ_dtheta * d2Z_ds2,
      dR_ds * d2R_dthetadzeta   + dR_dtheta * d2R_dsdzeta  + dZ_ds * d2Z_dthetadzeta  + dZ_dtheta * d2Z_dsdzeta, // d_i g_uw
      dR_ds * d2R_dtheta2       + dR_dtheta * d2R_dsdtheta + dZ_ds * d2Z_dtheta2      + dZ_dtheta * d2Z_dsdtheta, 
      2 * (R * dR_ds     + dR_dzeta * d2R_dsdzeta     + dZ_dzeta * d2Z_dsdzeta), 
      2 * (R * dR_dzeta  + dR_dzeta * d2R_dzeta2      + dZ_dzeta * d2Z_dzeta2),  // d_i g_vv
      2 * (R * dR_dtheta + dR_dzeta * d2R_dthetadzeta + dZ_dzeta * d2Z_dthetadzeta),  
      dR_dtheta * d2R_dsdzeta     + dR_dzeta * d2R_dsdtheta     + dZ_dtheta * d2Z_dsdzeta      + dZ_dzeta * d2Z_dsdtheta,
      dR_dtheta * d2R_dzeta2      + dR_dzeta * d2R_dthetadzeta  + dZ_dtheta * d2Z_dzeta2       + dZ_dzeta * d2Z_dthetadzeta, // d_i g_vw
      dR_dtheta * d2R_dthetadzeta + dR_dzeta * d2R_dtheta2      + dZ_dtheta * d2Z_dthetadzeta  + dZ_dzeta * d2Z_dtheta2,  
      2 * (dR_dtheta * d2R_dsdtheta     + dZ_dtheta * d2Z_dsdtheta), 
      2 * (dR_dtheta * d2R_dthetadzeta  + dZ_dtheta * d2Z_dthetadzeta), // d_i g_ww
      2 * (dR_dtheta * d2R_dtheta2      + dZ_dtheta * d2Z_dtheta2),  
  };
}

IR3 metric_vmec_interp3D::transform2cylindrical(const IR3& position) const {
  DenseVector x(3);
  x(0) = position[IR3::u];
  x(1) = this->reduce_phi(position[IR3::v]);
  x(2) = this->reduce_theta(position[IR3::w]);
  return {transform2cylindrical_spline_u_->eval(x),
          transform2cylindrical_spline_v_->eval(x),
          transform2cylindrical_spline_w_->eval(x)};
}

SM3 metric_vmec_interp3D::operator()(const IR3& position) const {
  DenseVector x(3);
  x(0) = position[IR3::u];
  x(1) = this->reduce_phi(position[IR3::v]);
  x(2) = this->reduce_theta(position[IR3::w]);
  return {metric_vmec_spline_uu_->eval(x),metric_vmec_spline_uv_->eval(x),metric_vmec_spline_uw_->eval(x),
          metric_vmec_spline_vv_->eval(x),metric_vmec_spline_vw_->eval(x),metric_vmec_spline_ww_->eval(x)};
}

dSM3 metric_vmec_interp3D::del(const IR3& position) const {
  DenseVector x(3);
  x(0) = position[IR3::u];
  x(1) = this->reduce_phi(position[IR3::v]);
  x(2) = this->reduce_theta(position[IR3::w]);
  auto del_metric_uu = metric_vmec_spline_uu_->evalJacobian(x);
  auto del_metric_uv = metric_vmec_spline_uv_->evalJacobian(x);
  auto del_metric_uw = metric_vmec_spline_uw_->evalJacobian(x);
  auto del_metric_vv = metric_vmec_spline_vv_->evalJacobian(x);
  auto del_metric_vw = metric_vmec_spline_vw_->evalJacobian(x);
  auto del_metric_ww = metric_vmec_spline_ww_->evalJacobian(x);
  return {del_metric_uu(0),del_metric_uu(1),del_metric_uu(2),
          del_metric_uv(0),del_metric_uv(1),del_metric_uv(2),
          del_metric_uw(0),del_metric_uw(1),del_metric_uw(2),
          del_metric_vv(0),del_metric_vv(1),del_metric_vv(2),
          del_metric_vw(0),del_metric_vw(1),del_metric_vw(2),
          del_metric_ww(0),del_metric_ww(1),del_metric_ww(2),
          };
}

//@todo move this to jacobian and think about testing this by calling the parent
double metric_vmec_interp3D::jacobian_vmec(const IR3& position) const {
  double s = position[IR3::u];
  double zeta = position[IR3::v];
  double theta = position[IR3::w];
  double J = 0.0;
//   #pragma omp parallel for reduction(+: J)
  for (size_t i = 0; i < xm_nyq_.size(); i++) {  
    J += (*gmnc_[i])(s) * std::cos( xm_nyq_[i]*theta - xn_nyq_[i]*zeta );
  };
  // left-handed VMEC coordinate system is re-oriented 
  // to u = Phi/Phi_bnd, v = zeta, w = theta for J>0
  // should we check/assume that signgs is always negative?
  return -J;
}
