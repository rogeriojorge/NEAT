// ::gyronimo:: - gyromotion for the people, by the people -
// An object-oriented library for gyromotion applications in plasma physics.
// Copyright (C) 2022 Jorge Ferreira and Paulo Rodrigues.

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

// @parser_boozxform.hh, this file is part of ::gyronimo::

#ifndef GYRONIMO_PARSER_BOOZXFORM
#define GYRONIMO_PARSER_BOOZXFORM

#include <string>
#include <netcdf>
#include <valarray>
#include <iostream>


namespace gyronimo {
//! Parsing object for `Boozxform` mapping files.
/*!
    Reads and parses a NetCDF file produced by `Boozxform`, ... 
*/
class parser_boozxform {
 public:
  typedef std::valarray<double> narray_type;
  parser_boozxform(const std::string& filename);
  ~parser_boozxform() {};
  bool lasym__logical() const {return lasym__logical__;}; //stellarator-symmetric boolean
  size_t nfp_b()           const {return nfp_b_;};  //Number of field periods
  size_t ns_b()            const {return ns_b_;};
  size_t nboz_b()          const {return nboz_b_;}; //Maximum toroidal mode number (divided by nfp) for *mnc/mns */
  size_t mboz_b()          const {return mboz_b_;}; //Maximum poloidal mode number for the input arrays *mnc/mns
  size_t mnboz_b()         const {return mnboz_b_;};
  // Dimensions
  size_t nradius()      const { return nradius_;};
  size_t radius_dim()   const { return radius_dim_;};
  size_t mn_mode()      const { return mn_mode_;};
  size_t mn_modes()     const { return mn_modes_;};
  size_t comput_surfs() const { return comput_surfs_;};
  size_t pack_rad()     const { return pack_rad_;};
  // Boozxform scalars
  double aspect_b()     const { return aspect_b_;};
  double rmax_b()       const { return rmax_b_;};
  double rmin_b()       const { return rmin_b_;};
  double betaxis_b()    const { return betaxis_b_;};
// Boozxform radial grids
  const narray_type& radius()        const { return radius_;};
// Boozxform profiles
  const narray_type& phi_b()      const { return phi_b_; };
  const narray_type& phip_b()     const { return phip_b_; };
  const narray_type& beta_b()     const { return beta_b_;};
  const narray_type& pres_b()     const { return pres_b_;};
  const narray_type& iota_b()    const { return iota_b_;};
  const narray_type& bvco_b()     const { return bvco_b_;};
  const narray_type& buco_b()     const { return buco_b_;};
  // Boozxform spectral representation
  const narray_type& jlist()       const { return jlist_;};
  const narray_type& ixm_b()       const { return ixm_b_;};
  const narray_type& ixn_b()       const { return ixn_b_;};
  const narray_type& rmnc_b()     const { return rmnc_b_;};
  const narray_type& rmns_b()     const { return rmns_b_;}; //This is 0 for stell sym
  const narray_type& zmns_b()     const { return zmns_b_; };
  const narray_type& zmnc_b()     const { return zmnc_b_; }; //This is 0 for stell sym
  const narray_type& pmns_b()     const { return pmns_b_; }; 
  const narray_type& pmnc_b()     const { return pmnc_b_; }; //This is 0 for stell sym
  const narray_type& gmnc_b()     const { return gmnc_b_; };
  const narray_type& gmns_b()     const { return gmns_b_; }; //This is 0 for stell sym
  const narray_type& bmnc_b()     const { return bmnc_b_; };
  const double B_0()              const { return bmnc_b_[0];};   //Additional stuff to review
  const narray_type& bmns_b()     const { return bmns_b_; }; //This is 0 for stell sym
  // const narray_type& bsubumnc() const { return bsubumnc_; };
  // const narray_type& bsubvmnc() const { return bsubvmnc_; };
  // const narray_type& bsubsmns() const { return bsubsmns_; };
  // const narray_type& bsubumnc() const { return bsubumns_; }; //This is 0 for stell sym
  // const narray_type& bsubvmnc() const { return bsubvmns_; }; //This is 0 for stell sym

  // in VMEC coordinates
  size_t ns_b_;
  bool lasym__logical__;

 private:
  size_t nfp_b_;
  size_t mn_mode_;
  size_t mn_modes_;
  size_t comput_surfs_;
  size_t pack_rad_;
  size_t version_;
  size_t nradius_;
  size_t radius_dim_;
  size_t nboz_b_; // size_t ntor_;
  size_t mboz_b_; // size_t mpol_;
  size_t mnboz_b_;// size_t mnmax_;
  double aspect_b_;
  double rmin_b_=0; //Not implemented in this version. Just 0.
  double rmax_b_=0; //Not implemented in this version. Just 0.
  double betaxis_b_=0; //Not implemented in this version. Just 0.
// VMEC profiles
  narray_type radius_;
  narray_type radius_half_;
  narray_type jlist_;
  narray_type ixm_b_;
  narray_type ixn_b_;
  narray_type iota_in_;
  narray_type iota_b_;
  narray_type bvco_in_, buco_in_;
  narray_type bvco_b_, buco_b_;
  narray_type pres_b_;
  narray_type beta_b_;
  narray_type phip_b_; 
  narray_type phi_b_;
  narray_type rmnc_b_;
  narray_type rmns_b_; // This is 0 for stell sym
  narray_type zmns_b_; 
  narray_type zmnc_b_; // This is 0 for stell sym
  narray_type pmns_b_;
  narray_type pmnc_b_;
  narray_type gmnc_b_;
  narray_type gmns_b_; // This is 0 for stell sym
  narray_type bmnc_b_;
  narray_type bmns_b_; // This is 0 for stell sym
  // narray_type bsubumnc_;
  // narray_type bsubvmnc_;
  // narray_type bsubsmns_;
  // narray_type bsubumns_;
  // narray_type bsubvmns_;
  void ERR(int , std::string);
  void getData(const netCDF::NcFile&, const std::string&, bool&);
  void getData(const netCDF::NcFile&, const std::string&, int&);
  void getData(const netCDF::NcFile&, const std::string&, size_t&);
  void getData(const netCDF::NcFile&, const std::string&, double&);
  void getData(const netCDF::NcFile&, const std::string&, narray_type&);
  void getData2D(const netCDF::NcFile&, const std::string&,  narray_type&);
  void getdim(const netCDF::NcFile&, const std::string, size_t& );
}; 

} // end namespace gyronimo.

#endif // GYRONIMO_parser_boozxform
