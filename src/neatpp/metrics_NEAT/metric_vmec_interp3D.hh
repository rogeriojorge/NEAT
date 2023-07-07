#include <gyronimo/parsers/parser_vmec.hh>
#include <gyronimo/metrics/metric_covariant.hh>
#include <gyronimo/interpolators/interpolator1d.hh>
#include <datatable.h>
#include <bspline.h>
#include <bsplinebuilder.h>

#ifndef GYRONIMO_METRIC_VMEC_INTERP3D
#define GYRONIMO_METRIC_VMEC_INTERP3D

using namespace gyronimo;
using namespace SPLINTER;

//! Covariant metric in `VMEC` curvilinear coordinates.
class metric_vmec_interp3D : public metric_covariant {
 public:
  typedef std::valarray<double> narray_type;
  typedef std::vector<interpolator1d> spectralarray_type;

  metric_vmec_interp3D(
    const parser_vmec *parser, const interpolator1d_factory *ifactory,
    const double *ns, const double *ntheta, const double *nzeta);
  virtual ~metric_vmec_interp3D() override;
  virtual SM3 metric_vmec(const IR3& position) const;
  virtual SM3 operator()(const IR3& position) const override;
  virtual dSM3 del(const IR3& position) const override;
  virtual IR3 transform2cylindrical(const IR3& position) const override;
  virtual IR3 transform2cylindrical_vmec(const IR3& position) const;
  virtual double jacobian_vmec(const IR3& position) const;
  virtual IR3 del_jacobian_vmec(const IR3& position) const;
  virtual double jacobian(const IR3& position) const override;
  virtual IR3 del_jacobian(const IR3& position) const override;
  const parser_vmec* parser() const {return parser_;};
  const double signgs() const {return signsgs_;};
  double reduce_theta(double theta) const;
  double reduce_phi(double phi) const;

 private:
  const parser_vmec* parser_;
  double b0_;
  const double phi_modulus_factor_;
  const double theta_modulus_factor_;
  int mnmax_, mnmax_nyq_, ns_, mpol_, ntor_, nfp_, signsgs_; 
  narray_type xm_, xn_, xm_nyq_, xn_nyq_; 
  interpolator1d **Rmnc_;
  interpolator1d **Zmns_;
  interpolator1d **gmnc_;

  BSpline* jacobian_spline_;

  BSpline* del_jacobian_spline_u_;
  BSpline* del_jacobian_spline_v_;
  BSpline* del_jacobian_spline_w_;

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
