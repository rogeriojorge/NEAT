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

// @parser_boozxform.cc, this file is part of ::gyronimo::

#include <cassert>
#include <gyronimo/core/error.hh>
#include <gyronimo/core/linspace.hh>
#include <gyronimo/parsers/parser_boozxform.hh>

namespace gyronimo {
  typedef std::valarray<double> narray_type;
  //! Reads and parses a Boozxform netcdf ouput file by name.
  parser_boozxform::parser_boozxform(const std::string& filename) {
    try {
      netCDF::NcFile dataFile(filename, netCDF::NcFile::read);
      getdim(dataFile, "radius", radius_dim_);
      getdim(dataFile, "mn_mode", mn_mode_);
      getdim(dataFile, "mn_modes", mn_modes_);
      getdim(dataFile, "comput_surfs", comput_surfs_);
      getdim(dataFile, "pack_rad", pack_rad_);
      getData(dataFile, "ns_b", ns_b_);
      
      nradius_ = ns_b_; 
      radius_ = linspace<narray_type>(0.0, 1.0, ns_b_-1);
      // radius_ = linspace<narray_type>(0.0, 1.0, pack_rad_);
      getData(dataFile, "nfp_b", nfp_b_);
      getData(dataFile, "nboz_b", nboz_b_);
      getData(dataFile, "mboz_b", mboz_b_);
      getData(dataFile, "mnboz_b", mnboz_b_);
      getData(dataFile, "aspect_b", aspect_b_);
      getData(dataFile, "rmin_b", rmin_b_);
      getData(dataFile, "rmax_b", rmax_b_);
      getData(dataFile, "betaxis_b", betaxis_b_);
      getData(dataFile, "lasym__logical__", lasym__logical__);
      getData(dataFile, "phi_b", phi_b_);
      getData(dataFile, "phip_b", phip_b_);
      getData(dataFile, "pres_b", pres_b_);
      getData(dataFile, "iota_b", iota_in_);
      getData(dataFile, "beta_b", beta_b_);
      getData(dataFile, "bvco_b", bvco_in_);
      getData(dataFile, "buco_b", buco_in_);
      getData(dataFile, "jlist", jlist_);

      iota_b_.resize(radius_dim_-1);
      bvco_b_.resize(radius_dim_-1);
      buco_b_.resize(radius_dim_-1);
      for (int j = 0; j < (radius_dim_-1); j++) {
        iota_b_[j] = iota_in_[j+1];
        bvco_b_[j] = bvco_in_[j+1];
        buco_b_[j] = buco_in_[j+1];
      }

      getData(dataFile, "ixm_b", ixm_b_);
      getData(dataFile, "ixn_b", ixn_b_);
      getData2D(dataFile, "rmnc_b", rmnc_b_);
      getData2D(dataFile, "zmns_b", zmns_b_);
      getData2D(dataFile, "pmns_b", pmns_b_);
      getData2D(dataFile, "gmn_b", gmnc_b_);
      getData2D(dataFile, "bmnc_b", bmnc_b_);
      
      if (lasym__logical__) {
        getData2D(dataFile, "rmns_b", rmns_b_);
        getData2D(dataFile, "zmnc_b", zmnc_b_);
        getData2D(dataFile, "pmnc_b", pmnc_b_);
        getData2D(dataFile, "gmns_b", gmns_b_);
        getData2D(dataFile, "bmns_b", bmns_b_);

      } else  {
          // Stellarator-symmetric.

        bmns_b_.resize(0, 0);
        rmns_b_.resize(0, 0);
        zmnc_b_.resize(0, 0);
        pmnc_b_.resize(0, 0);
        gmns_b_.resize(0, 0);
      }
      // getData2D(dataFile, "bsubumnc", bsubumnc_);
      // getData2D(dataFile, "bsubvmnc", bsubvmnc_);
      // getData2D(dataFile, "bsubsmns", bsubsmns_);

    } catch (netCDF::exceptions::NcException &e) {
      std::cout << e.what() << std::endl;
    }
  }

  void parser_boozxform::ERR(int e, std::string details) {
      std::cout << "NetCDF Error! " << nc_strerror(e)
                << " [" << details << "]" << std::endl;
      throw std::runtime_error(nc_strerror(e));
  } 
  void parser_boozxform::getData(const netCDF::NcFile& nc, const std::string& var, bool& out){
      auto data = nc.getVar(var); 
      if(data.isNull()) error(__func__, __FILE__, __LINE__,
                        ("Null data for var: "+var).c_str() , 1);
      assert( data.getDimCount() == 0 );
      int dataIn; data.getVar(&dataIn);
      out = ( dataIn % 2 ? true : false );
  }
  void parser_boozxform::getData(const netCDF::NcFile& nc, const std::string& var, int& out){
      auto data = nc.getVar(var); 
      if(data.isNull()) error(__func__, __FILE__, __LINE__,
                  ("Null data for var: "+var).c_str() , 1);assert( data.getDimCount() == 0 );
      data.getVar(&out);
  }
  void parser_boozxform::getData(const netCDF::NcFile& nc, const std::string& var, size_t& out){
      auto data = nc.getVar(var); 
      if(data.isNull()) error(__func__, __FILE__, __LINE__,
                  ("Null data for var: "+var).c_str() , 1);
      assert( data.getDimCount() == 0 );
      int dataIn; data.getVar(&dataIn);
      out = (size_t) dataIn;
  }
  void parser_boozxform::getData(const netCDF::NcFile& nc, const std::string& var, double& out){
      auto data = nc.getVar(var); 
      if(data.isNull()) error(__func__, __FILE__, __LINE__,
                  ("Null data for var: "+var).c_str() , 1);
      assert( data.getDimCount() == 0 );
      data.getVar(&out);
  }
  void parser_boozxform::getData(const netCDF::NcFile& nc, const std::string& var,  narray_type& out){
      auto data = nc.getVar(var); 
      if(data.isNull()) error(__func__, __FILE__, __LINE__,
                  ("Null data for var: "+var).c_str() , 1);
      assert( data.getDimCount() == 1 );
      auto dims = data.getDims(); size_t size  = dims[0].getSize();
      out.resize(size);
      data.getVar(&out[0]);
  }
  void parser_boozxform::getData2D(const netCDF::NcFile& nc, const std::string& var,  narray_type& out){
      auto data = nc.getVar(var); 
      if(data.isNull()) error(__func__, __FILE__, __LINE__,
                  ("Null data for var: "+var).c_str() , 1);
      assert( data.getDimCount() == 2 );
      auto dims = data.getDims(); 
      size_t size0  = dims[0].getSize(); size_t size1  = dims[1].getSize();
      out.resize(size0*size1);
      data.getVar(&out[0]);
  }
  void parser_boozxform::getdim(const netCDF::NcFile& nc, const std::string var, size_t& out) {
      int dim_id, retval;
      size_t dimval;
      int ncid=nc.getId();
      if ((retval = nc_inq_dimid(ncid, var.c_str(), &dim_id)))
        ERR(retval, var);
      if ((retval = nc_inq_dimlen(ncid, dim_id, &dimval)))
        ERR(retval, var);
      out = (size_t) dimval;
}
} // end namespace gyronimo.
