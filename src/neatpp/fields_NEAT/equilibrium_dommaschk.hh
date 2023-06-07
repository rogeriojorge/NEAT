// ::gyronimo:: - gyromotion for the people, by the people -
// An object-oriented library for gyromotion applications in plasma physics.
// Copyright (C) 2021 Miguel Pereira.

// ::gyronimo:: is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// ::gyronimo:: is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with ::gyronimo::.  If not, see <https://www.gnu.org/licenses/>.


#ifndef GYRONIMO_EQUILIBRIUM_DOMMASCHK
#define GYRONIMO_EQUILIBRIUM_DOMMASCHK

#include <gyronimo/fields/IR3field_c1.hh>
#include <metric_cylindrical.hh>
#include <gyronimo/core/dblock.hh>


namespace gyronimo{

//! Tokamak equilibrium magnetic field in 'HELENA' curvilinear coordinates.
/*!
    Following IR3Field rules, the magnetic field is normalised by a `m_factor`
    matching its on-axis value `B0()` in [T], which is located at `R0()` in [m].
    The coordinates are set by the `metric_helena` object and the type of 2d
    interpolators is set by the specific `interpolator2d_factory` supplied.
    Contravariant components have dimensions of [m^{-1}]. Being an
    **equilibrium** field, `t_factor` is set to one.
    
    Only the minimal interface is implemented for the moment and further
    specialisations may enhance the object's performance.
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

  
 /* equilibrium_dommaschk operator+(equilibrium_dommaschk const& obj)
    {
    equilibrium_dommaschk result(const metric_cylindrical *g,1,1,1,1,1); //placeholder values just to initialise

    IR3 res::contravariant(const IR3& position, double time) const override {
        return {
	       obj.contravariant(position, time)+contravariant(position, time);
        };

    IR3 res::del_contravariant(const IR3& position, double time) const override {
        return {
	       obj.del_contravariant(position, time)+del_contravariant(position, time);
        };
    }

    }
     //   res.setValueB(BR1() + obj.BR1(), Bphi1() + obj.Bphi1(), BZ1() + obj.BZ1());
    return result;
    };*/

/*
void setValueB(double B1, double B2, double B3)  const
  {
    BR1_=B1;
    Bphi1_=B2;
    BZ1_=B3;

  }*/
  /*
  void setValueB()  
  {
    BR1_=1;
    Bphi1_=3;
    BZ1_=2;

  }*/
  const int m() const {return m_;};
  const int l() const {return l_;};
  const double coeff1() const {return coeff1_;};
  const double coeff2() const {return coeff2_;};
  const double B0() const {return B0_;};
 // const double BR1() const {return BR1_;};
  //const double Bphi1() const {return Bphi1_;};
  //const double BZ1() const {return BZ1_;};
  
  


 private:
  const metric_cylindrical *metric_;
  int m_;
  int l_;
  double coeff1_;
  double coeff2_;
  double B0_;
  double BR1_;
  double Bphi1_;
  double BZ1_;
  int flag;
  double dB[9];
  
};

}// end namespace gyronimo.


#endif // GYRONIMO_EQUILIBRIUM_DOMMASCHK
