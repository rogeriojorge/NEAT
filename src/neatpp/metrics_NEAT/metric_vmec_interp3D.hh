#ifndef GYRONIMO_METRIC_VMEC_INTERP3D
#define GYRONIMO_METRIC_VMEC_INTERP3D

#include <gyronimo/parsers/parser_vmec.hh>
#include <gyronimo/metrics/metric_covariant.hh>
#include <gyronimo/interpolators/interpolator1d.hh>

namespace gyronimo{

//! Covariant metric in `VMEC` curvilinear coordinates.
class metric_vmec_interp3D : public metric_covariant {
 public:
  typedef std::valarray<double> narray_type;
  typedef std::vector<interpolator1d> spectralarray_type;

  metric_vmec_interp3D(
    const parser_vmec *parser, const interpolator1d_factory *ifactory);
  virtual ~metric_vmec_interp3D() override;
  virtual SM3 operator()(const IR3& position) const override;
  virtual dSM3 del(const IR3& position) const override;
  virtual IR3 transform2cylindrical(const IR3& position) const override;
  double jacobian_vmec(const IR3& position) const;
  const parser_vmec* parser() const {return parser_;};
  const double signgs() const {return signsgs_;};

 private:
  const parser_vmec* parser_;
  double b0_;
  int mnmax_, mnmax_nyq_, ns_, mpol_, ntor_, nfp_, signsgs_; 
  narray_type xm_, xn_, xm_nyq_, xn_nyq_; 
  interpolator1d **Rmnc_;
  interpolator1d **Zmns_;
  interpolator1d **gmnc_;
};

} // end namespace gyronimo

#endif // GYRONIMO_METRIC_VMEC
