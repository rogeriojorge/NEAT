#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include "neatpp.hh"

// Python wrapper functions
PYBIND11_MODULE(neatpp, m) {
    m.doc() = "Gyronimo Wrapper for the Stellarator Near-Axis Expansion (STELLNA)";
    m.def("gc_solver",                      &gc_solver);
    m.def("gc_solver_ensemble",             &gc_solver_ensemble);
    m.def("gc_solver_qs",                   &gc_solver_qs);
    m.def("gc_solver_qs_ensemble",          &gc_solver_qs_ensemble);
    m.def("gc_solver_qs_partial",           &gc_solver_qs_partial);
    m.def("gc_solver_qs_partial_ensemble",  &gc_solver_qs_partial_ensemble);
}
