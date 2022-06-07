#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include "neatpp.hh"

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
}
