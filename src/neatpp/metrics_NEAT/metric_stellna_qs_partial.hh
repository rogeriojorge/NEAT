// ::gyronimo:: - gyromotion for the people, by the people -
// An object-oriented library for gyromotion applications in plasma physics.
// Copyright (C) 2022 Rogerio Jorge.

// @metric_stellna_qs_partial.hh

#ifndef METRIC_STELLNA_QS_PARTIAL
#define METRIC_STELLNA_QS_PARTIAL

#include <gyronimo/metrics/metric_covariant.hh>
#include <gyronimo/interpolators/interpolator1d.hh>
using namespace gyronimo;

//! Covariant metric in stellarator near-axis coordinates.
/*!
    In the near-axis expansion, the field is provided directly
    in Boozer coordinates, both the covariant and contravariant
    forms so that the metric is unnecessary
*/
class metric_stellna_qs_partial : public metric_covariant {
 public:
  metric_stellna_qs_partial(int field_periods, double Bref,
    const dblock& phi_grid, double G0, double G2,
    double I2, double iota, double iotaN,
    double B0,  double B1c,
    const dblock& B20, double B2c, double beta1s,
    const interpolator1d_factory* ifactory);
  virtual ~metric_stellna_qs_partial() override;

  virtual SM3 operator()(const IR3& position) const override;
  virtual dSM3 del(const IR3& position) const override;
  virtual double jacobian(const IR3& position) const override;
  virtual IR3 del_jacobian(const IR3& position) const override;
  double reduce_phi(double phi) const;
  const double B0() const {return B0_;};
  const double B1c() const {return B1c_;};
  const interpolator1d* B20() const {return B20_;};
  const double B2c() const {return B2c_;};
  const double beta1s() const {return beta1s_;};
  const double Bref() const {return Bref_;};
  const double iota() const {return iota_;};
  const double iotaN() const {return iotaN_;};
  const double G0() const {return G0_;};
  const double G2() const {return G2_;};
  const double I2() const {return I2_;};
  const double field_periods() const {return field_periods_;};

  private:
    const double phi_modulus_factor_;
    double Bref_, G0_, G2_, I2_, iota_, iotaN_;
    double B0_, B1c_, B2c_, beta1s_;
    double field_periods_;
    interpolator1d *B20_;
};

#endif // METRIC_STELLNA_QS_PARTIAL