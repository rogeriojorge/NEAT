// ::gyronimo:: - gyromotion for the people, by the people -
// An object-oriented library for gyromotion applications in plasma physics.

#ifndef GYRONIMO_EQUILIBRIUM_DOMMASCHK
#define GYRONIMO_EQUILIBRIUM_DOMMASCHK

#include <gyronimo/fields/IR3field_c1.hh>
#include <metric_cylindrical.hh>
#include <gyronimo/core/dblock.hh>


namespace gyronimo{

/* Stellarator equilibrium magnetic field using Dommaschk potentials, in cylindrical coordinates
    To initialize an equilibrium_dommaschk object such that it can be used, 
    a metric_cylindrical object must be provided, along with a magnetic field
    normalization B0. Additionally, these potentials are characterized by four values.
    They are: m and l, two integers that represent the number of toroidal and poloidal
    periods of the field. The other two numbers are coeff1 and coeff2, two doubles.
    If one is following the 1986 paper by W. Dommaschk, these values represent the following:
    when l is even, coeff1 and coeff2 correspond to b and c of equation (12),
    and if l is odd, then coeff1 and coeff2 correspond to a and d of equation (12).
    This simplification assumed the Stellarator symmetry V_{m,l}(R,\phi,Z) = -V_{m,l}(R,-\phi,-Z)

*/

/*
Possible opimizations: not adding the 1/R and -1/R² terms in each field to then subtract
them, instead just using the equilibrium_inverse_R_factor defined in dommaschktrace.hh to add
those terms once at the end
Also, there is currently no parallel computing 
*/
class equilibrium_dommaschk : public IR3field_c1{
 public:
  equilibrium_dommaschk(
      const metric_cylindrical *g, int m, int l, double coeff1, double coeff2, double B0);
  virtual ~equilibrium_dommaschk() override {};

  virtual IR3 contravariant(const IR3& position, double time) const override;
  virtual dIR3 del_contravariant(
      const IR3& position, double time) const override;
  virtual IR3 partial_t_contravariant(
      const IR3& position, double time) const override {return {0.0,0.0,0.0};};

  const int m() const {return m_;};
  const int l() const {return l_;};
  const double coeff1() const {return coeff1_;};
  const double coeff2() const {return coeff2_;};
  const double B0() const {return B0_;};

 private:
  const metric_cylindrical *metric_;
  int m_;
  int l_;
  double coeff1_;
  double coeff2_;
  double B0_;  
};

}// end namespace gyronimo.


#endif // GYRONIMO_EQUILIBRIUM_DOMMASCHK
