#ifndef GYRONIMO_EQUILIBRIUM_VMEC_INTERP3D
#define GYRONIMO_EQUILIBRIUM_VMEC_INTERP3D

#include <gyronimo/fields/IR3field_c1.hh>
#include "metric_vmec_interp3D.hh"
#include <gyronimo/parsers/parser_vmec.hh>
#include <gyronimo/interpolators/interpolator2d.hh>
// #include <datatable.h>
// #include <bspline.h>
// #include <bsplinebuilder.h>

// using namespace gyronimo;
// using namespace SPLINTER;

class equilibrium_vmec_interp3D : public IR3field_c1{
 public:
  typedef std::valarray<double> narray_type;
  typedef std::vector<interpolator1d> spectralarray_type;
  equilibrium_vmec_interp3D(
      const metric_vmec_interp3D *g, const interpolator1d_factory *ifactory);
  virtual ~equilibrium_vmec_interp3D() override;

  virtual IR3 contravariant_vmec(const IR3& position, double time) const;
  virtual dIR3 del_contravariant_vmec(const IR3& position, double time) const;
  virtual IR3 contravariant(const IR3& position, double time) const override;
  virtual dIR3 del_contravariant(const IR3& position, double time) const override;
  virtual IR3 partial_t_contravariant(
      const IR3& position, double time) const override {return {0.0,0.0,0.0};};
  double magnitude_vmec(const IR3& position, double time) const;

  double R_0() const {return metric_->parser()->R_0();};
  double B_0() const {return metric_->parser()->B_0();};
  
  const metric_vmec_interp3D* metric() const {return metric_;};
 private:
  const metric_vmec_interp3D *metric_;
  narray_type xm_nyq_, xn_nyq_; 
  interpolator1d **bmnc_;
  interpolator1d **bsupumnc_;
  interpolator1d **bsupvmnc_;

//   BSpline* contravariant_vmec_spline_u_;
  BSpline* contravariant_vmec_spline_v_;
  BSpline* contravariant_vmec_spline_w_;

//   BSpline* del_contravariant_vmec_spline_uu_;
//   BSpline* del_contravariant_vmec_spline_uv_;
//   BSpline* del_contravariant_vmec_spline_uw_;
//   BSpline* del_contravariant_vmec_spline_vu_;
//   BSpline* del_contravariant_vmec_spline_vv_;
//   BSpline* del_contravariant_vmec_spline_vw_;
//   BSpline* del_contravariant_vmec_spline_wu_;
//   BSpline* del_contravariant_vmec_spline_wv_;
//   BSpline* del_contravariant_vmec_spline_ww_;
};

#endif // GYRONIMO_EQUILIBRIUM_VMEC
