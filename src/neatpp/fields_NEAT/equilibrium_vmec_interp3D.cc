#include <gyronimo/core/dblock.hh>
#include "equilibrium_vmec_interp3D.hh"
// #include <iostream>

using namespace gyronimo;
// using namespace SPLINTER;

equilibrium_vmec_interp3D::equilibrium_vmec_interp3D(
    const metric_vmec_interp3D *g, const interpolator1d_factory *ifactory)
    : IR3field_c1(abs(g->parser()->B_0()), 1.0, g), metric_(g),
      xm_nyq_(g->parser()->xm_nyq()), xn_nyq_(g->parser()->xn_nyq()),   
      bmnc_(nullptr), bsupumnc_(nullptr), bsupvmnc_(nullptr) {
  const parser_vmec *p = metric_->parser();
  dblock_adapter s_range(p->radius());
  dblock_adapter s_half_range(p->radius_half());
  // set spectral interpolators 
  bmnc_ = new interpolator1d* [xm_nyq_.size()];
  bsupumnc_ = new interpolator1d* [xm_nyq_.size()];
  bsupvmnc_ = new interpolator1d* [xm_nyq_.size()];
//@todo NEED TO FIX AXIS AND EDGE! TBI! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//   #pragma omp parallel for
  for(size_t i=0; i<xm_nyq_.size(); i++) {
    std::slice s_cut (i, s_range.size(), xm_nyq_.size());
    std::valarray<double> bsupumnc_i = (p->bsupumnc())[s_cut] / this->m_factor();
    bsupumnc_[i] = ifactory->interpolate_data( s_range, dblock_adapter(bsupumnc_i));
    std::valarray<double> bsupvmnc_i = (p->bsupvmnc())[s_cut] / this->m_factor();
    bsupvmnc_[i] = ifactory->interpolate_data( s_range, dblock_adapter(bsupvmnc_i));
    // bmnc is defined a half radius
    std::slice s_h_cut (i+xm_nyq_.size(), s_half_range.size(), xm_nyq_.size());
    std::valarray<double> bmnc_i = (p->bmnc())[s_h_cut] / this->m_factor();
    bmnc_[i] = ifactory->interpolate_data( s_half_range, dblock_adapter(bmnc_i));
  };
    DataTable contravariant_vmec_samples_u, contravariant_vmec_samples_v, contravariant_vmec_samples_w;
    DataTable del_contravariant_vmec_samples_uu, del_contravariant_vmec_samples_uv, del_contravariant_vmec_samples_uw,
              del_contravariant_vmec_samples_vu, del_contravariant_vmec_samples_vv, del_contravariant_vmec_samples_vw,
              del_contravariant_vmec_samples_wu, del_contravariant_vmec_samples_wv, del_contravariant_vmec_samples_ww;
    IR3 contravariant_vmec_temp = {0, 0, 0};
    dIR3 del_contravariant_vmec_temp = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    // dblock_adapter s_radius = (p->radius());
    auto s_min = s_range[0];
    auto s_max = s_range[s_range.size() - 1];
    auto ds = (s_max - s_min) / (metric_->ns_interp() - 1);
    auto dtheta = metric_->theta_modulus_factor() / metric_->ntheta_interp();
    auto dzeta = metric_->phi_modulus_factor() / metric_->nzeta_interp();
    DenseVector x(3);
    IR3 pos = {0,0,0};

    for (size_t i = 0; i < metric_->ns_interp(); ++i) {
        for (size_t j = 0; j <= metric_->ntheta_interp(); ++j) {
            for (size_t k = 0; k <= metric_->nzeta_interp(); ++k) {
                x(0) = s_min + i * ds;
                x(1) = k * dzeta;
                x(2) = j * dtheta;
                pos = {x(0), x(1), x(2)};
                
                contravariant_vmec_temp = contravariant_vmec(pos, 0);
                // contravariant_vmec_samples_u.addSample(x, contravariant_vmec_temp[IR3::u]);
                contravariant_vmec_samples_v.addSample(x, contravariant_vmec_temp[IR3::v]);
                contravariant_vmec_samples_w.addSample(x, contravariant_vmec_temp[IR3::w]);

                // del_contravariant_vmec_temp = del_contravariant_vmec(pos, 0);
                // del_contravariant_vmec_samples_uu.addSample(x, del_contravariant_vmec_temp[dIR3::uu]);
                // del_contravariant_vmec_samples_uv.addSample(x, del_contravariant_vmec_temp[dIR3::uv]);
                // del_contravariant_vmec_samples_uw.addSample(x, del_contravariant_vmec_temp[dIR3::uw]);
                // del_contravariant_vmec_samples_vu.addSample(x, del_contravariant_vmec_temp[dIR3::vu]);
                // del_contravariant_vmec_samples_vv.addSample(x, del_contravariant_vmec_temp[dIR3::vv]);
                // del_contravariant_vmec_samples_vw.addSample(x, del_contravariant_vmec_temp[dIR3::vw]);
                // del_contravariant_vmec_samples_wu.addSample(x, del_contravariant_vmec_temp[dIR3::wu]);
                // del_contravariant_vmec_samples_wv.addSample(x, del_contravariant_vmec_temp[dIR3::wv]);
                // del_contravariant_vmec_samples_ww.addSample(x, del_contravariant_vmec_temp[dIR3::ww]);
            }
        }
    }
    // contravariant_vmec_spline_u_ = new BSpline(BSpline::Builder(contravariant_vmec_samples_u).degree(1).build());
    contravariant_vmec_spline_v_ = new BSpline(BSpline::Builder(contravariant_vmec_samples_v).degree(1).build());
    contravariant_vmec_spline_w_ = new BSpline(BSpline::Builder(contravariant_vmec_samples_w).degree(1).build());

    // del_contravariant_vmec_spline_uu_ = new BSpline(BSpline::Builder(del_contravariant_vmec_samples_uu).degree(3).build());
    // del_contravariant_vmec_spline_uv_ = new BSpline(BSpline::Builder(del_contravariant_vmec_samples_uv).degree(3).build());
    // del_contravariant_vmec_spline_uw_ = new BSpline(BSpline::Builder(del_contravariant_vmec_samples_uw).degree(3).build());
    // del_contravariant_vmec_spline_vu_ = new BSpline(BSpline::Builder(del_contravariant_vmec_samples_vu).degree(3).build());
    // del_contravariant_vmec_spline_vv_ = new BSpline(BSpline::Builder(del_contravariant_vmec_samples_vv).degree(3).build());
    // del_contravariant_vmec_spline_vw_ = new BSpline(BSpline::Builder(del_contravariant_vmec_samples_vw).degree(3).build());
    // del_contravariant_vmec_spline_wu_ = new BSpline(BSpline::Builder(del_contravariant_vmec_samples_wu).degree(3).build());
    // del_contravariant_vmec_spline_wv_ = new BSpline(BSpline::Builder(del_contravariant_vmec_samples_wv).degree(3).build());
    // del_contravariant_vmec_spline_ww_ = new BSpline(BSpline::Builder(del_contravariant_vmec_samples_ww).degree(3).build());
}
equilibrium_vmec_interp3D::~equilibrium_vmec_interp3D() {
  if(bmnc_) delete bmnc_;
  if(bsupumnc_) delete bsupumnc_;
  if(bsupvmnc_) delete bsupvmnc_;

  // if(contravariant_vmec_spline_u_) delete contravariant_vmec_spline_u_;
  if(contravariant_vmec_spline_v_) delete contravariant_vmec_spline_v_;
  if(contravariant_vmec_spline_w_) delete contravariant_vmec_spline_w_;

  // if(del_contravariant_vmec_spline_uu_) delete del_contravariant_vmec_spline_uu_;
  // if(del_contravariant_vmec_spline_uv_) delete del_contravariant_vmec_spline_uv_;
  // if(del_contravariant_vmec_spline_uw_) delete del_contravariant_vmec_spline_uw_;
  // if(del_contravariant_vmec_spline_vu_) delete del_contravariant_vmec_spline_vu_;
  // if(del_contravariant_vmec_spline_vv_) delete del_contravariant_vmec_spline_vv_;
  // if(del_contravariant_vmec_spline_vw_) delete del_contravariant_vmec_spline_vw_;
  // if(del_contravariant_vmec_spline_wu_) delete del_contravariant_vmec_spline_wu_;
  // if(del_contravariant_vmec_spline_wv_) delete del_contravariant_vmec_spline_wv_;
  // if(del_contravariant_vmec_spline_ww_) delete del_contravariant_vmec_spline_ww_;
}

IR3 equilibrium_vmec_interp3D::contravariant_vmec(const IR3& position, double time) const {
  double s = position[IR3::u];
  double zeta = position[IR3::v];
  double theta = position[IR3::w];
  double B_theta = 0.0, B_zeta = 0.0;
//   #pragma omp parallel for reduction(+: B_zeta, B_theta)
  for (size_t i = 0; i < xm_nyq_.size(); i++) {  
    double m = xm_nyq_[i]; double n = xn_nyq_[i];
    double cosmn = std::cos( m*theta - n*zeta );
    B_zeta += (*bsupvmnc_[i])(s) * cosmn;
    B_theta += (*bsupumnc_[i])(s) * cosmn; 
  };
  return {0.0,  B_zeta, B_theta};
}
dIR3 equilibrium_vmec_interp3D::del_contravariant_vmec(
    const IR3& position, double time) const {
  double s = position[IR3::u];
  double zeta = position[IR3::v];
  double theta = position[IR3::w];
  double B_theta = 0.0, B_zeta = 0.0;
  double dB_theta_ds = 0.0, dB_theta_dtheta = 0.0, dB_theta_dzeta = 0.0;
  double dB_zeta_ds = 0.0, dB_zeta_dtheta = 0.0, dB_zeta_dzeta = 0.0;
//   #pragma omp parallel for reduction(+: B_zeta, B_theta, dB_theta_ds, dB_theta_dtheta, dB_theta_dzeta, dB_zeta_ds, dB_zeta_dtheta, dB_zeta_dzeta)
  for (size_t i = 0; i < xm_nyq_.size(); i++) {  
    double m = xm_nyq_[i]; double n = xn_nyq_[i];
    double cosmn = std::cos( m*theta - n*zeta );
    double sinmn = std::sin( m*theta - n*zeta );
    double bsupumnc_i = (*bsupumnc_[i])(s);
    double bsupvmnc_i = (*bsupvmnc_[i])(s);
    B_theta += bsupumnc_i * cosmn; 
    B_zeta += bsupvmnc_i * cosmn;
    dB_theta_ds += (*bsupumnc_[i]).derivative(s) * cosmn;
    dB_theta_dtheta -= m * bsupumnc_i * sinmn;
    dB_theta_dzeta += n * bsupumnc_i * sinmn;
    dB_zeta_ds += (*bsupvmnc_[i]).derivative(s) * cosmn;
    dB_zeta_dtheta -= m * bsupvmnc_i * sinmn;
    dB_zeta_dzeta += n * bsupvmnc_i * sinmn;
  };
  return {
      0.0, 0.0, 0.0, 
	    dB_zeta_ds, dB_zeta_dzeta, dB_zeta_dtheta,
      dB_theta_ds, dB_theta_dzeta, dB_theta_dtheta
  };              
}

IR3 equilibrium_vmec_interp3D::contravariant(const IR3& position, double time) const {
  DenseVector x(3);
  x(0) = position[IR3::u];
  x(1) = metric_->reduce_phi(position[IR3::v]);
  x(2) = metric_->reduce_theta(position[IR3::w]);
  IR3 contravariant_temp = {0,
                            contravariant_vmec_spline_v_->eval(x),
                            contravariant_vmec_spline_w_->eval(x)};
  return contravariant_temp;
}

dIR3 equilibrium_vmec_interp3D::del_contravariant(const IR3& position, double time) const {
  DenseVector x(3);
  x(0) = position[IR3::u];
  x(1) = metric_->reduce_phi(position[IR3::v]);
  x(2) = metric_->reduce_theta(position[IR3::w]);
  // auto del_contravariant_u = contravariant_vmec_spline_u_->evalJacobian(x);
  auto del_contravariant_v = contravariant_vmec_spline_v_->evalJacobian(x);
  auto del_contravariant_w = contravariant_vmec_spline_w_->evalJacobian(x);
  return {0,0,0,
          del_contravariant_v(0), del_contravariant_v(1), del_contravariant_v(2),
          del_contravariant_w(0), del_contravariant_w(1), del_contravariant_w(2)};   
}

//@todo we can actually override the methods to calculate the covariant components of the field
//@todo move this to magnitude after the half radius issue is sorted out
double equilibrium_vmec_interp3D::magnitude_vmec(
    const IR3& position, double time) const {
  double s = position[IR3::u];
  double zeta = position[IR3::v];
  double theta = position[IR3::w];
  double Bnorm = 0.0;
//   #pragma omp parallel for reduction(+: Bnorm)
  for (size_t i = 0; i < xm_nyq_.size(); i++) {  
    Bnorm += (*bmnc_[i])(s) * std::cos( xm_nyq_[i] * theta - xn_nyq_[i] *zeta );
  };
  return Bnorm;
}
