#include "metric_boozxform.hh"
#include <numbers>
#include <gyronimo/core/error.hh>
#include <numbers>
using namespace std;

namespace gyronimo{

metric_boozxform::metric_boozxform(
    const parser_boozxform *p, const interpolator1d_factory *ifactory) 
    : parser_(p), b0_(p->B_0()), mnboz_b_(p->mnboz_b()),
      ns_b_(p->ns_b()), ixm_b_(p->ixm_b()), ixn_b_(p->ixn_b()), 
      Rmnc_b_(nullptr), Zmns_b_(nullptr), gmnc_b_(nullptr),
      psi_boundary_(p->phi_b()[p->phi_b().size()-1]/2/numbers::pi), numns_b_(nullptr)
    {
    dblock_adapter s_range(p->radius());
    Rmnc_b_ = new interpolator1d* [ixm_b_.size()];
    Zmns_b_ = new interpolator1d* [ixm_b_.size()];
    gmnc_b_ = new interpolator1d* [ixm_b_.size()];
    numns_b_ = new interpolator1d* [ixm_b_.size()];
    valarray<double> rmnc = p->rmnc_b();
    valarray<double> zmns = p->zmns_b();
    valarray<double> gmnc = psi_boundary_*(p->gmnc_b());
    valarray<double> numns = p->pmns_b();
    // #pragma omp parallel for
    for(size_t i=0; i<ixm_b_.size(); i++) {
      slice s_cut (i, s_range.size(), ixm_b_.size());
      valarray<double> rmnc_i = rmnc[s_cut];
      Rmnc_b_[i] = ifactory->interpolate_data( s_range, dblock_adapter(rmnc_i));
      valarray<double> zmnc_i = zmns[s_cut];
      Zmns_b_[i] = ifactory->interpolate_data( s_range, dblock_adapter(zmnc_i));
      valarray<double> gmnc_i = gmnc[s_cut]; // note the psi_boundary_ factor above
      gmnc_b_[i] = ifactory->interpolate_data( s_range, dblock_adapter(gmnc_i));
      valarray<double> numns_i = numns[s_cut];
      numns_b_[i] = ifactory->interpolate_data( s_range, dblock_adapter(numns_i));
    };
}
metric_boozxform::~metric_boozxform() {
  if(Rmnc_b_) delete Rmnc_b_;
  if(Zmns_b_) delete Zmns_b_;
  if(gmnc_b_) delete gmnc_b_;
  if(numns_b_) delete numns_b_;
}
SM3 metric_boozxform::operator()(const IR3& position) const {
  error(__func__, __FILE__, __LINE__, "Explicit metric not available.", 1);
  return {0, 0, 0, 0, 0, 0};
}
dSM3 metric_boozxform::del(const IR3& position) const {
  error(__func__, __FILE__, __LINE__, "Explicit metric not available.", 1);
  return {
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
      0, 0, 0};
}
IR3 metric_boozxform::transform2cylindrical(const IR3& position) const {
    //What we call cylindrical phi0 is the cylindrical coordinate on axis
    double s = position[IR3::u];
    double theta = numbers::pi-position[IR3::v];
    double zeta = position[IR3::w];
    double R = 0.0, Z = 0.0, m = 0.0, n = 0.0, Nu=0.0;
  
    // #pragma omp parallel for reduction(+: R, Z)
    for (size_t i = 0; i<ixm_b_.size(); i++) {
      m = ixm_b_[i]; n = ixn_b_[i];
      R+= (*Rmnc_b_[i])(s) * cos( m*theta - n*zeta ); 
      Z+= (*Zmns_b_[i])(s) * sin( m*theta - n*zeta );
      Nu+= (*numns_b_[i])(s) * sin( m*theta - n*zeta );
    }
    double zeta0 = zeta - Nu;
    return  {R, zeta0, Z};
}

double metric_boozxform::jacobian(const IR3& position) const {
  double s = position[IR3::u];
  double theta = numbers::pi-position[IR3::v];
  double zeta = position[IR3::w];
  double J = 0.0;
  // #pragma omp parallel for reduction(+: J)
  for (size_t i = 0; i < ixm_b_.size(); i++) {  
    J += (*gmnc_b_[i])(s) * cos( ixm_b_[i]*theta - ixn_b_[i]*zeta );
  };
  return J;
}
IR3 metric_boozxform::del_jacobian(const IR3& position) const {
  double s = position[IR3::u];
  double theta = numbers::pi-position[IR3::v];
  double zeta = position[IR3::w];
  double J_ds = 0.0, J_dzeta = 0.0, J_dtheta = 0.0, sintheta = 0.0, gmnc = 0.0;
  // #pragma omp parallel for reduction(+: J_ds, J_dzeta, J_dtheta )
  for (size_t i = 0; i < ixm_b_.size(); i++) {  
    J_ds     += (*gmnc_b_[i]).derivative(s)  * cos( ixm_b_[i]*theta - ixn_b_[i]*zeta );
    sintheta = sin( ixm_b_[i]*theta - ixn_b_[i]*zeta );
    gmnc = (*gmnc_b_[i])(s);
    J_dtheta += ixm_b_[i] * gmnc * sintheta;
    J_dzeta  += ixn_b_[i] * gmnc * sintheta;
  };
  return {J_ds, J_dtheta, J_dzeta};
}
} // end namespace gyronimo
