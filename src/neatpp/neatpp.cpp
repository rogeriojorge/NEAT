#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include "stellna.hh"
#include "vmectrace.hh"
#include "vmec_loss.hh"
#include "booztrace.hh"

namespace py = pybind11;

// Python wrapper functions
PYBIND11_MODULE(neatpp, m) {
      m.doc() = "Python Wrapper for the gyronimo-based C++ functions that traces"
                  "particle orbits in a stellarator magnetic field (near-axis or VMEC)";
      m.def("gc_solver", &gc_solver,
            "Trace a single particle in a fully quasisymmetric near-axis magnetic field");
      m.def("gc_solver_ensemble", &gc_solver_ensemble,
            "Trace a particle ensemble in a fully quasisymmetric near-axis magnetic field");
      m.def("gc_solver_qs", &gc_solver_qs,
            "Trace a single particle in a general near-axis magnetic field");
      m.def("gc_solver_qs_ensemble",&gc_solver_qs_ensemble,
            "Trace a particle ensemble in a general quasisymmetric near-axis magnetic field");
      m.def("gc_solver_qs_partial",&gc_solver_qs_partial,
            "Trace a single particle in a partially quasisymmetric near-axis magnetic field");
      m.def("gc_solver_qs_partial_ensemble", &gc_solver_qs_partial_ensemble,
            "Trace a particle ensemble in a partially quasisymmetric near-axis magnetic field");
      m.def("vmectrace",&vmectrace,
            "Trace a single particle in a VMEC equilibrium magnetic field",
            py::arg("vmec_file"), py::arg("integrator"), py::arg("maximum_s"), py::arg("charge"), py::arg("mass"),
            py::arg("Lambda"), py::arg("vpp_sign"), py::arg("energy"),
            py::arg("s0"), py::arg("theta0"), py::arg("phi0"),
            py::arg("Tfinal"), py::arg("nsamples"));
      m.def("vmecloss",&vmecloss,
            "Trace an ensemble of particles in a VMEC equilibrium magnetic field",
            py::arg("vmec_file"), py::arg("integrator"), py::arg("maximum_s"), py::arg("charge"), py::arg("mass"),
            py::arg("energy"), py::arg("nlambda_trapped"), py::arg("nlambda_passing"),
            py::arg("r0"), py::arg("r_max"), py::arg("ntheta"),
            py::arg("nphi"), py::arg("nsamples"),py::arg("Tfinal"), py::arg("nthreads"), py::arg("dist"));
      m.def("booztrace",&booztrace,
            "Trace a single particle in a Boozxform equilibrium magnetic field",
            py::arg("vmec_file"), py::arg("integrator"), py::arg("maximum_s"), py::arg("charge"), py::arg("mass"),
            py::arg("Lambda"), py::arg("vpp_sign"), py::arg("energy"),
            py::arg("s0"), py::arg("theta0"), py::arg("phi0"),
            py::arg("Tfinal"), py::arg("nsamples"));
}
