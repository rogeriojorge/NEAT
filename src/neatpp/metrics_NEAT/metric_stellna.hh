// ::gyronimo:: - gyromotion for the people, by the people -
// An object-oriented library for gyromotion applications in plasma physics.
// Copyright (C) 2021 Paulo Rodrigues, Rogerio Jorge.

// @metric_stellna.hh

#ifndef GYRONIMO_METRIC_STELLNA
#define GYRONIMO_METRIC_STELLNA

#include <gyronimo/metrics/metric_covariant.hh>
#include <gyronimo/interpolators/interpolator1d.hh>
using namespace gyronimo;

//! Covariant metric in stellarator near-axis coordinates.
/*!
    In the near-axis expansion, the field is provided directly
    in Boozer coordinates, both the covariant and contravariant
    forms so that the metric is unnecessary
*/
class metric_stellna : public metric_covariant {
 public:
  metric_stellna(int field_periods, double Bref,
    const dblock& phi_grid, double G0, double G2,
    double I2, double iota, double iotaN,
    const dblock& B0,  const dblock& B1c, const dblock& B1s,
    const dblock& B20, const dblock& B2c, const dblock& B2s,
    const dblock& beta0,  const dblock& beta1c, const dblock& beta1s,
    const interpolator1d_factory* ifactory);
  virtual ~metric_stellna() override;

  virtual SM3 operator()(const IR3& position) const override;
  virtual dSM3 del(const IR3& position) const override;
  virtual double jacobian(const IR3& position) const override;
  virtual IR3 del_jacobian(const IR3& position) const override;
  double reduce_phi(double phi) const;
  const interpolator1d* B0() const {return B0_;};
  const interpolator1d* B1c() const {return B1c_;};
  const interpolator1d* B1s() const {return B1s_;};
  const interpolator1d* B20() const {return B20_;};
  const interpolator1d* B2c() const {return B2c_;};
  const interpolator1d* B2s() const {return B2s_;};
  const interpolator1d* beta0() const {return beta0_;};
  const interpolator1d* beta1c() const {return beta1c_;};
  const interpolator1d* beta1s() const {return beta1s_;};
  const double Bref() const {return Bref_;};
  const double iota() const {return iota_;};
  const double iotaN() const {return iotaN_;};
  const double G0() const {return G0_;};
  const double G2() const {return G2_;};
  const double I2() const {return I2_;};
  const double field_periods() const {return field_periods_;};

  private:
    const double phi_modulus_factor_;
    double Bref_, G0_, G2_, I2_, iota_, iotaN_, field_periods_, phi0_;
    interpolator1d *B0_, *B1c_, *B1s_, *B20_, *B2c_, *B2s_, *beta0_, *beta1c_, *beta1s_;
};

#endif // GYRONIMO_METRIC_STELLNA