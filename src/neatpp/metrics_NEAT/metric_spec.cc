// SPEC has coordinates (s in [-1,1], theta in [0, 2 pi], zeta in [0, 2pi])
// R_cylindrical = ((1+s)/2)*R_mn^out + ((1-s)/2)*R_mn^in
// s=-1 is a coordinate axis, s=1 is the surface
// For tokamak/stellarator physics (toroidal geometry), use igeometry = 3
// For the implementation of metric tensor: SPEC matlabtools get_spec_metric
// See the equation for the metric in SPEC_manual.pdf equation 40

// Magnetic field contravariant: SPEC matlabtools get_spec magfield
// See the equation for the magnetic field in SPEC_manual.pdf equation 4
// Stellarator symmetry simplified the calculations