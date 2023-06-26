#include "metric_vmec_interp3D.hh"

namespace gyronimo{

metric_vmec_interp3D::metric_vmec_interp3D(
    const parser_vmec *p, const interpolator1d_factory *ifactory) 
    : parser_(p), b0_(p->B_0()), mnmax_(p->mnmax()), mnmax_nyq_(p->mnmax_nyq()),
      ns_(p->ns()), mpol_(p->mpol()), ntor_(p->ntor()), 
      signsgs_(p->signgs()), nfp_(p->nfp()),
      xm_(p->xm()), xn_(p->xn()), xm_nyq_(p->xm_nyq()), xn_nyq_(p->xn_nyq()),
      Rmnc_(nullptr), Zmns_(nullptr), gmnc_(nullptr)
      {
    // set radial grid block
    dblock_adapter s_range(p->radius());
    dblock_adapter s_half_range(p->radius_half());
    // set spectral components 
    Rmnc_ = new interpolator1d* [xm_.size()];
    Zmns_ = new interpolator1d* [xm_.size()];
    gmnc_ = new interpolator1d* [xm_.size()];
//@todo NEED TO FIX AXIS AND EDGE! TBI! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #pragma omp parallel for
    for(size_t i=0; i<xm_.size(); i++) {
      std::slice s_cut (i, s_range.size(), xm_.size());
      std::valarray<double> rmnc_i = (p->rmnc())[s_cut];
      Rmnc_[i] = ifactory->interpolate_data( s_range, dblock_adapter(rmnc_i));
      std::valarray<double> zmnc_i = (p->zmns())[s_cut];
      Zmns_[i] = ifactory->interpolate_data( s_range, dblock_adapter(zmnc_i));
      // note that gmnc is defined at half mesh
      std::slice s_h_cut (i+xm_nyq_.size(), s_half_range.size(), xm_nyq_.size());
      std::valarray<double> gmnc_i = (p->gmnc())[s_h_cut];
      gmnc_[i] = ifactory->interpolate_data( s_half_range, dblock_adapter(gmnc_i));
    };
}
metric_vmec_interp3D::~metric_vmec_interp3D() {
  if(Rmnc_) delete Rmnc_;
  if(Zmns_) delete Zmns_;
  if(gmnc_) delete gmnc_;
}
SM3 metric_vmec_interp3D::operator()(const IR3& position) const {
  double s = position[IR3::u];
  double zeta = position[IR3::v];
  double theta = position[IR3::w];
  double R = 0.0, dR_ds = 0.0, dR_dtheta = 0.0, dR_dzeta = 0.0;
  double Z = 0.0, dZ_ds = 0.0, dZ_dtheta = 0.0, dZ_dzeta = 0.0;

   #pragma omp parallel for reduction(+: R, Z, dR_ds, dR_dtheta, dR_dzeta, dZ_ds, dZ_dtheta, dZ_dzeta)
  for (size_t i = 0; i<xm_.size(); i++) {  
    double m = xm_[i]; double n = xn_[i];
    double cosmn = std::cos( m*theta - n*zeta );
    double sinmn = std::sin( m*theta - n*zeta );
    double rmnc_i = (*Rmnc_[i])(s); 
    double zmns_i = (*Zmns_[i])(s);
    // assuming for now that vmec equilibrium has stellarator symmetry.
    R += rmnc_i * cosmn; 
    Z += zmns_i * sinmn;
    dR_ds += (*Rmnc_[i]).derivative(s) * cosmn; 
    dR_dtheta -= m * rmnc_i * sinmn; 
    dR_dzeta += n * rmnc_i * sinmn;
    dZ_ds += (*Zmns_[i]).derivative(s) * sinmn; 
    dZ_dtheta += m * zmns_i * cosmn; 
    dZ_dzeta -= n * zmns_i * cosmn; 
  };
  return {
    dR_ds * dR_ds + dZ_ds * dZ_ds,                      // g_uu
    dR_ds * dR_dzeta + dZ_ds * dZ_dzeta,                // g_uw
    dR_ds * dR_dtheta + dZ_ds * dZ_dtheta,              // g_uv
    R * R + dR_dzeta * dR_dzeta + dZ_dzeta * dZ_dzeta,  // g_vv
    dR_dtheta * dR_dzeta + dZ_dtheta * dZ_dzeta,        // g_vw
    dR_dtheta * dR_dtheta + dZ_dtheta * dZ_dtheta       // g_ww
  };
}
dSM3 metric_vmec_interp3D::del(const IR3& position) const {
  double s = position[IR3::u];
  double zeta = position[IR3::v];
  double theta = position[IR3::w];
  double R = 0.0, Z = 0.0;
  double dR_ds = 0.0,        dR_dtheta = 0.0,       dR_dzeta = 0.0;
  double d2R_ds2 = 0.0,      d2R_dsdtheta = 0.0,    d2R_dsdzeta = 0.0;
  double d2R_dthetads = 0.0, d2R_dtheta2 = 0.0,     d2R_dthetadzeta = 0.0; 
  double d2R_dzetads = 0.0,  d2R_dzetadtheta = 0.0, d2R_dzeta2 = 0.0;
  double dZ_ds = 0.0,        dZ_dtheta = 0.0,       dZ_dzeta = 0.0;
  double d2Z_ds2 = 0.0,      d2Z_dsdtheta = 0.0,    d2Z_dsdzeta = 0.0;
  double d2Z_dthetads = 0.0, d2Z_dtheta2 = 0.0,     d2Z_dthetadzeta = 0.0; 
  double d2Z_dzetads = 0.0,  d2Z_dzetadtheta = 0.0, d2Z_dzeta2 = 0.0;

  #pragma omp parallel for reduction(+: R, dR_ds, dR_dtheta, dR_dzeta, d2R_ds2, d2R_dsdtheta, d2R_dsdzeta, d2R_dtheta2, d2R_dthetadzeta, d2R_dzeta2, Z, dZ_ds ,dZ_dtheta, dZ_dzeta, d2Z_ds2, d2Z_dsdtheta, d2Z_dsdzeta, d2Z_dtheta2, d2Z_dthetadzeta, d2Z_dzeta2)
  for (size_t i = 0; i<xm_.size(); i++) {  
    double m = xm_[i]; double n = xn_[i];
    double cosmn = std::cos( m*theta - n*zeta );
    double sinmn = std::sin( m*theta - n*zeta );
    double rmnc_i = (*Rmnc_[i])(s); 
    double zmns_i = (*Zmns_[i])(s);
    double d_rmnc_i = (*Rmnc_[i]).derivative(s); 
    double d_zmns_i = (*Zmns_[i]).derivative(s); 
    double d2_rmnc_i = (*Rmnc_[i]).derivative2(s);
    double d2_zmns_i = (*Zmns_[i]).derivative2(s);
    // assuming for now that vmec equilibrium has stellarator symmetry.
    R += rmnc_i * cosmn; Z += zmns_i * sinmn;
    dR_ds += d_rmnc_i * cosmn; 
    dR_dtheta -= m * rmnc_i * sinmn; 
    dR_dzeta += n * rmnc_i * sinmn;
    d2R_ds2 += d2_rmnc_i * cosmn; 
    d2R_dsdtheta -= m * d_rmnc_i * sinmn;
    d2R_dsdzeta += n * d_rmnc_i * sinmn;
    d2R_dtheta2 -= m * m * rmnc_i * cosmn;
    d2R_dthetadzeta += m * n * rmnc_i * cosmn;
    d2R_dzeta2 -= n * n * rmnc_i * cosmn;
    dZ_ds += d_zmns_i * sinmn; 
    dZ_dtheta += m * zmns_i * cosmn; 
    dZ_dzeta -= n * zmns_i * cosmn; 
    d2Z_ds2 += d2_zmns_i * sinmn;
    d2Z_dsdtheta += m * d_zmns_i * cosmn;
    d2Z_dsdzeta -= n * d_zmns_i * cosmn;
    d2Z_dtheta2 -= m * m * zmns_i * sinmn;
    d2Z_dthetadzeta += m * n * zmns_i * sinmn;
    d2Z_dzeta2 -= n * n * zmns_i * sinmn;
}
//@todo still need to test this carefully. Find a way to test d_g!
  return {
      2 * (dR_ds * d2R_ds2      + dZ_ds * d2Z_ds2), 
      2 * (dR_ds * d2R_dsdzeta  + dZ_ds * d2Z_dsdzeta), // d_i g_uu
      2 * (dR_ds * d2R_dsdtheta + dZ_ds * d2Z_dsdtheta), 
      dR_ds * d2R_dsdzeta       + dR_dzeta * d2R_ds2      + dZ_ds * d2Z_dsdzeta      + dZ_dzeta * d2Z_ds2,
      dR_ds * d2R_dzeta2        + dR_dzeta * d2R_dsdzeta  + dZ_ds * d2Z_dzeta2       + dZ_dzeta * d2Z_dsdzeta,// d_i g_uv
      dR_ds * d2R_dthetadzeta   + dR_dzeta * d2R_dsdtheta  + dZ_ds * d2Z_dthetadzeta  + dZ_dzeta * d2Z_dsdtheta, 
      dR_ds * d2R_dsdtheta      + dR_dtheta * d2R_ds2      + dZ_ds * d2Z_dsdtheta     + dZ_dtheta * d2Z_ds2,
      dR_ds * d2R_dthetadzeta   + dR_dtheta * d2R_dsdzeta  + dZ_ds * d2Z_dthetadzeta  + dZ_dtheta * d2Z_dsdzeta, // d_i g_uw
      dR_ds * d2R_dtheta2       + dR_dtheta * d2R_dsdtheta + dZ_ds * d2Z_dtheta2      + dZ_dtheta * d2Z_dsdtheta, 
      2 * (R * dR_ds     + dR_dzeta * d2R_dsdzeta     + dZ_dzeta * d2Z_dsdzeta), 
      2 * (R * dR_dzeta  + dR_dzeta * d2R_dzeta2      + dZ_dzeta * d2Z_dzeta2),  // d_i g_vv
      2 * (R * dR_dtheta + dR_dzeta * d2R_dthetadzeta + dZ_dzeta * d2Z_dthetadzeta),  
      dR_dtheta * d2R_dsdzeta     + dR_dzeta * d2R_dsdtheta     + dZ_dtheta * d2Z_dsdzeta      + dZ_dzeta * d2Z_dsdtheta,
      dR_dtheta * d2R_dzeta2      + dR_dzeta * d2R_dthetadzeta  + dZ_dtheta * d2Z_dzeta2       + dZ_dzeta * d2Z_dthetadzeta, // d_i g_vw
      dR_dtheta * d2R_dthetadzeta + dR_dzeta * d2R_dtheta2      + dZ_dtheta * d2Z_dthetadzeta  + dZ_dzeta * d2Z_dtheta2,  
      2 * (dR_dtheta * d2R_dsdtheta     + dZ_dtheta * d2Z_dsdtheta), 
      2 * (dR_dtheta * d2R_dthetadzeta  + dZ_dtheta * d2Z_dthetadzeta), // d_i g_ww
      2 * (dR_dtheta * d2R_dtheta2      + dZ_dtheta * d2Z_dtheta2),  
  };
}

IR3 metric_vmec_interp3D::transform2cylindrical(const IR3& position) const {
    double u = position[gyronimo::IR3::u];
    double v = position[gyronimo::IR3::v];
    double w = position[gyronimo::IR3::w];
    double R = 0.0, Z = 0.0;
  
    #pragma omp parallel for reduction(+: R, Z)
    for (size_t i = 0; i<xm_.size(); i++) {
      double m = xm_[i]; double n = xn_[i];
      R+= (*Rmnc_[i])(u) * std::cos( m*w - n*v ); 
      Z+= (*Zmns_[i])(u) * std::sin( m*w - n*v );
    }
    return  {R, v, Z};
}
 
double metric_vmec_interp3D::jacobian_vmec(const IR3& position) const {
  double s = position[IR3::u];
  double zeta = position[IR3::v];
  double theta = position[IR3::w];
  double J = 0.0;
  #pragma omp parallel for reduction(+: J)
  for (size_t i = 0; i < xm_nyq_.size(); i++) {  
    J += (*gmnc_[i])(s) * std::cos( xm_nyq_[i]*theta - xn_nyq_[i]*zeta );
  };
  return -J;
}

IR3 metric_vmec_interp3D::del_jacobian_vmec(const IR3& position) const {
  double s = position[IR3::u];
  double zeta = position[IR3::v];
  double theta = position[IR3::w];
  double dJds = 0.0;
  #pragma omp parallel for reduction(+: dJds)
  for (size_t i = 0; i < xm_nyq_.size(); i++) {  
    dJds += (*gmnc_[i]).derivative(s) * std::cos( xm_nyq_[i]*theta - xn_nyq_[i]*zeta );
  };
  double dJdtheta = 0.0;
  #pragma omp parallel for reduction(+: dJdtheta)
  for (size_t i = 0; i < xm_nyq_.size(); i++) {  
    dJdtheta += -(*gmnc_[i])(s) * std::sin( xm_nyq_[i]*theta - xn_nyq_[i]*zeta )*(xm_nyq_[i]);
  };
  double dJdzeta = 0.0;
  #pragma omp parallel for reduction(+: dJdzeta)
  for (size_t i = 0; i < xm_nyq_.size(); i++) {  
    dJdzeta += -(*gmnc_[i])(s) * std::sin( xm_nyq_[i]*theta - xn_nyq_[i]*zeta )*(-xn_nyq_[i]);
  };
  return {-dJds, -dJdtheta, -dJdzeta};
}

void metric_vmec_interp3D::precompute_jacobian_grid(double *grid, size_t ns, size_t ntheta, size_t nzeta) const {
    double ds = 1.0 / (ns - 1);
    double dtheta = 2 * M_PI / (ntheta - 1);
    double dzeta = 2 * M_PI / (nzeta - 1);

    #pragma omp parallel for
    for (size_t is = 0; is < ns; ++is) {
        for (size_t itheta = 0; itheta < ntheta; ++itheta) {
            for (size_t izeta = 0; izeta < nzeta; ++izeta) {
                double s = ds * is;
                double theta = dtheta * itheta;
                double zeta = dzeta * izeta;
                IR3 position = {s, theta, zeta};
                size_t index = izeta + nzeta * (itheta + ntheta * is);
                grid[index] = this->jacobian_vmec(position);
            }
        }
    }
}
void interpolate_jacobian(const double *grid, double s, double theta, double zeta,
                          size_t ns, size_t ntheta, size_t nzeta, double *result) {
    const double x[1] = {s};
    const double y[1] = {theta};
    const double z[1] = {zeta};

    // Call the splinterp parallel_interp3 function to interpolate
    splinterp::parallel_interp3(splinterp::interp3_F<double>, grid, ns, ntheta, nzeta, x, y, z, 1, result, 1);
}

double metric_vmec_interp3D::jacobian(const IR3& position) const {
    double s = position[IR3::u];
    double zeta = position[IR3::v];
    double theta = position[IR3::w];
    double J = 0.0;
    // Precompute the grid
    size_t ns = 100; // Number of s grid points
    size_t ntheta = 100; // Number of theta grid points
    size_t nzeta = 100; // Number of zeta grid points
    double *grid = new double[ns * ntheta * nzeta];
    precompute_jacobian_grid(grid, ns, ntheta, nzeta);

    // Interpolate for an arbitrary point
    double result;
    interpolate_jacobian(grid, s, theta, zeta, ns, ntheta, nzeta, &result);

    std::cout << "Interpolated value: " << result << std::endl;

    delete[] grid;
    return result;
}

// // Not able to get CMakeLists SPLINTER to share library with NEAT
// #include <datatable.h>
// #include <bspline.h>
// #include <bsplinebuilder.h>
// using namespace SPLINTER;

// double metric_vmec_interp3D::jacobian_vmec_interp3d(const IR3& position) const {
//     double s = position[IR3::u];
//     double zeta = position[IR3::v];
//     double theta = position[IR3::w];

//     // Create new DataTable to manage samples
//     DataTable samples;
    
//     // Sample the function
//     DenseVector x(3);  // Initialize DenseVector with size 3
//     for (size_t i = 0; i < xm_nyq_.size(); i++)
//     {
//         x(0) = s;       // Use parentheses for element access
//         x(1) = theta;   // Use parentheses for element access
//         x(2) = zeta;    // Use parentheses for element access
//         double y = (*gmnc_[i])(s) * std::cos( xm_nyq_[i]*theta - xn_nyq_[i]*zeta );
//         samples.addSample(x, y);
//     }

//     // Build B-splines that interpolate the samples
//     BSpline bspline = BSpline::Builder(samples).degree(3).build();

//     // Evaluate the approximant at (s, theta, zeta)
//     x(0) = s;        // Use parentheses for element access
//     x(1) = theta;    // Use parentheses for element access
//     x(2) = zeta;     // Use parentheses for element access

//     // Return the interpolated value
//     return -bspline.eval(x);
// }


}