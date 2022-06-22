### setup project ###
cmake_minimum_required(VERSION 3.17.3) # 3.17 > for Python3_SOABI
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(neatf90
  VERSION 1.0
  DESCRIPTION "neatf90 module"
  LANGUAGES C CXX Fortran
  )

# # Safety net
# if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
#   message(
#     FATAL_ERROR
#       "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.\n"
#   )
# endif()

# Load netcdf and openmp
find_program(NF_CONFIG "nf-config")
if (NF_CONFIG)
execute_process(COMMAND "nf-config" --prefix
        OUTPUT_VARIABLE NFPREFIX)
else()
message(SEND_ERROR "nf-config not found. Please install libnetcdff-dev")
endif()
string(STRIP ${NFPREFIX} NFPREFIX)
set(NFINC ${NFPREFIX}/include)
set(NFLIBS ${NFPREFIX}/lib)
include_directories ($ENV{NETCDFF_INCLUDE} ${NFINC})
link_directories ($ENV{NETCDF_LIB} $ENV{NETCDFF_LIB} ${NFLIBS})
set(CMAKE_Fortran_FLAGS "-cpp -std=f2008 -Wall -Wno-unused -Wno-unused-dummy-argument -fopenmp")

# Grab Python
find_package(Python 3 REQUIRED
  COMPONENTS Interpreter Development NumPy)
find_package(PythonLibs REQUIRED)

# Grab the variables from a local Python installation
# F2PY headers
execute_process(
  COMMAND "${Python_EXECUTABLE}"
  -c "import numpy.f2py; print(numpy.f2py.get_include())"
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Project scope; consider using target_include_directories instead
include_directories(
  BEFORE
  ${Python_INCLUDE_DIRS}
  ${Python_NumPy_INCLUDE_DIRS}
  ${F2PY_INCLUDE_DIR}
  )

message(STATUS ${Python_INCLUDE_DIRS})
message(STATUS ${F2PY_INCLUDE_DIR})
message(STATUS ${Python_NumPy_INCLUDE_DIRS})

# Vars
set(f2py_module_name "neatf90")
set(fortran_pyf_file "${PROJECT_SOURCE_DIR}/neatf90.pyf")
set(fortran_src_file "${PROJECT_SOURCE_DIR}/neatf90.f90")
set(f2py_module_c "${f2py_module_name}module.c")
set(generated_module_file "${f2py_module_name}${Python_SOABI}")

# Generate sources
add_custom_target(
  genpyf
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_c}"
  )
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_c}" "${f2py_module_name}-f2pywrappers2.f90"
  COMMAND ${Python_EXECUTABLE}  -m "numpy.f2py"
                   "${fortran_pyf_file}"
                   -m "${f2py_module_name}"
                   --lower # Important
  DEPENDS ${fortran_pyf_file} # Fortran source
  )

if (NOT TARGET simple)
    set(SIMPLE_LOCATION "${PROJECT_SOURCE_DIR}/../../external/simple/SRC")
    add_library (simple SHARED
      ${SIMPLE_LOCATION}/canonical_coordinates_mod.f90
      ${SIMPLE_LOCATION}/nctools_module.f90
      ${SIMPLE_LOCATION}/odeint_rkf45.f90
      ${SIMPLE_LOCATION}/contrib/minpack.f90
      ${SIMPLE_LOCATION}/magfie.f90
      ${SIMPLE_LOCATION}/boozer_converter.f90
      ${SIMPLE_LOCATION}/binsrc.f90
      ${SIMPLE_LOCATION}/plag_coeff.f90
      ${SIMPLE_LOCATION}/chamb_m.f90
      ${SIMPLE_LOCATION}/sub_alpha_lifetime_can.f90
      ${SIMPLE_LOCATION}/vmecinm_m.f90
      ${SIMPLE_LOCATION}/spline_vmec_data.f90
      ${SIMPLE_LOCATION}/spl_three_to_five.f90
      ${SIMPLE_LOCATION}/new_vmec_allocation_stuff.f90
      ${SIMPLE_LOCATION}/get_canonical_coordinates.f90
      ${SIMPLE_LOCATION}/testing.f90
      ${SIMPLE_LOCATION}/field_can.f90
      ${SIMPLE_LOCATION}/orbit_symplectic.f90
      ${SIMPLE_LOCATION}/orbit_symplectic_quasi.f90
      ${SIMPLE_LOCATION}/common.f90
      ${SIMPLE_LOCATION}/simple.f90
      ${SIMPLE_LOCATION}/bench.f90
      ${SIMPLE_LOCATION}/parse_ants.f90
      ${SIMPLE_LOCATION}/zzg.f90
      ${SIMPLE_LOCATION}/sorting.f90
      ${SIMPLE_LOCATION}/check_orbit_type.f90
    )
    add_library ( rkf45lib SHARED
      ${SIMPLE_LOCATION}/contrib/rkf45.f90
    )
    target_link_libraries(simple rkf45lib netcdf netcdff lapack blas)
    target_link_libraries(simple gomp)
    # target_link_libraries(genpyf simple)
endif()

# Set up target
set(neatf90_library_name neatf90)
add_library(${neatf90_library_name} SHARED
  "${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_c}" # Generated
  "${F2PY_INCLUDE_DIR}/fortranobject.c" # From NumPy
  "${fortran_src_file}" # Fortran source(s)
  )


# Depend on sources
add_dependencies(${neatf90_library_name} genpyf)
# target_link_libraries(${neatf90_library_name} PRIVATE ${Python_LIBRARIES} simple)
target_link_libraries(${neatf90_library_name} PRIVATE simple)
target_link_libraries(${neatf90_library_name} PRIVATE ${Python_LIBRARIES})
set_target_properties(
     ${neatf90_library_name}
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME "${neatf90_library_name}"
        LINKER_LANGUAGE C
        SUFFIX ".so"
    )
