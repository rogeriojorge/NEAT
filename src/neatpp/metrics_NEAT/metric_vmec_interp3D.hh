#ifndef GYRONIMO_METRIC_VMEC_INTERP3D
#define GYRONIMO_METRIC_VMEC_INTERP3D

#include <gyronimo/parsers/parser_vmec.hh>
#include <gyronimo/metrics/metric_covariant.hh>
#include <gyronimo/interpolators/interpolator1d.hh>
#include <datatable.h>
#include <bspline.h>
#include <bsplinebuilder.h>
#include "linterp.h"

// add example similar to https://github.com/rncarpio/linterp/blob/master/src/linterp_test2.cpp

// add ALGLIB for spline interpolation

using namespace gyronimo;
using namespace SPLINTER;

class metric_vmec_interp3D : public metric_covariant {
 public:
  typedef std::valarray<double> narray_type;
  typedef std::vector<interpolator1d> spectralarray_type;

  metric_vmec_interp3D(
    const parser_vmec *parser, const interpolator1d_factory *ifactory,
    double ns_interp_input, double ntheta_interp_input, double nzeta_interp_input);
  virtual ~metric_vmec_interp3D() override;
  virtual SM3 operator()(const IR3& position) const override;
  virtual dSM3 del(const IR3& position) const override;
  virtual IR3 transform2cylindrical(const IR3& position) const override;
  virtual IR3 transform2cylindrical_vmec(const IR3& position) const;
  virtual SM3 metric_vmec(const IR3& position) const;
  virtual dSM3 del_metric_vmec(const IR3& position) const;
  double jacobian_vmec(const IR3& position) const;
  const parser_vmec* parser() const {return parser_;};
  const double signgs() const {return signsgs_;};
  double reduce_theta(double theta) const;
  double reduce_phi(double phi) const;
  double ns_interp() const {return ns_interp_;};
  double ntheta_interp() const {return ntheta_interp_;};
  double nzeta_interp() const {return nzeta_interp_;};
  double theta_modulus_factor() const {return theta_modulus_factor_;};
  double phi_modulus_factor() const {return phi_modulus_factor_;};

 private:
  const parser_vmec* parser_;
  double b0_;
  const double phi_modulus_factor_, theta_modulus_factor_;
  int mnmax_, mnmax_nyq_, ns_, mpol_, ntor_, nfp_, signsgs_; 
  double ns_interp_,ntheta_interp_,nzeta_interp_;
  narray_type xm_, xn_, xm_nyq_, xn_nyq_; 
  interpolator1d **Rmnc_;
  interpolator1d **Zmns_;
  interpolator1d **gmnc_;

  BSpline* transform2cylindrical_spline_u_;
  BSpline* transform2cylindrical_spline_v_;
  BSpline* transform2cylindrical_spline_w_;

  BSpline* metric_vmec_spline_uu_;
  BSpline* metric_vmec_spline_uv_;
  BSpline* metric_vmec_spline_uw_;
  BSpline* metric_vmec_spline_vv_;
  BSpline* metric_vmec_spline_vw_;
  BSpline* metric_vmec_spline_ww_;
};

#endif // GYRONIMO_METRIC_VMEC
