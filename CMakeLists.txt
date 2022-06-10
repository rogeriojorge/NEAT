cmake_minimum_required(VERSION 3.5)
project(neatpp
  LANGUAGES C CXX Fortran
)

# Enable output of compile commands during generation.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Add the -fPIC flag using the line below
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Use python version that satisfies version constraints
set(Python_FIND_STRATEGY LOCATION)
# Make sure that python libraries such as numpy are found
find_package(Python 3 COMPONENTS Interpreter Development.Module NumPy)
# Use relative paths in MacOS
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_LIBDIR lib)
# detect virtualenv and set Pip args accordingly
# if(DEFINED ENV{VIRTUAL_ENV} OR DEFINED ENV{CONDA_PREFIX})
#   set(_pip_args)
# else()
#   set(_pip_args "--user")
# endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Logging messages to debug possible Python issues
find_program(PYTHON "python3")
message(status " Python executable is ${PYTHON}")
message(status " Python Development Module found value is ${Python_Development.Module_FOUND}")
message(status " Python version is ${Python_VERSION}")

# Download git submodules
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

add_compile_options(-DOPENMP)
find_package(OpenMP)

# Add BOOST
find_package(Boost 1.71.0)
if(Boost_FOUND)
    message(STATUS "Boost version is ${Boost_VERSION_STRING}")
    message(STATUS "Boost include dirs are ${Boost_INCLUDE_DIRS}")
else()
    message(STATUS "Downloading and installing boost.")
    # For some external project macros
    include(ExternalProject)

    # Download boost from git and build the headers
    set( boost_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/boost )
    set( boost_target boost)

    ExternalProject_Add(
        ${boost_target}
        PREFIX ${boost_DIR}
        GIT_REPOSITORY https://github.com/boostorg/boost.git
        GIT_TAG boost-1.76.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        GIT_SUBMODULES tools/build tools/boost_install libs/config libs/numeric
            libs/math libs/type_traits libs/predef libs/assert libs/static_assert
            libs/throw_exception libs/core libs/serialization libs/preprocessor libs/mpl
            libs/utility libs/typeof libs/array libs/units libs/integer libs/fusion
            libs/range libs/iterator libs/concept_check libs/detail libs/function_types
            libs/lexical_cast libs/container libs/move libs/smart_ptr libs/multi_array
            libs/functional libs/function libs/type_index libs/container_hash libs/bind
        CONFIGURE_COMMAND ./bootstrap.sh --prefix=<PREFIX>
        BUILD_COMMAND ./b2 headers --prefix=${boost_DIR}
        BUILD_IN_SOURCE 1
        INSTALL_COMMAND ""
    )

    set(Boost_INCLUDE_DIRS ${boost_DIR}/src/${boost_target})
    message(STATUS "Boost include dirs are ${Boost_INCLUDE_DIRS}")
endif()
include_directories(${Boost_INCLUDE_DIRS})

# Add GSL
find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})
# Compile git submodules
if(INSTALL_SIMPLE)
    add_subdirectory(external/simple)
    # Install fffi
    configure_file("${PROJECT_SOURCE_DIR}/src/neat/grammar.tx" 
    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/neat/grammar.tx" COPYONLY)
    # execute_process(
    #     COMMAND pip show fffi
    #     RESULT_VARIABLE EXIT_CODE
    #     OUTPUT_QUIET
    # )
    # if (NOT ${EXIT_CODE} EQUAL 0)
        # set(TARGET_NAME fffi)

        # file(GLOB pyfiles ${PROJECT_SOURCE_DIR}/external/fffi/fffi/*.py)
        # file(GLOB pyfiles_parser ${PROJECT_SOURCE_DIR}/external/fffi/fffi/parser/*)
        # foreach (filename ${pyfiles})
        #     get_filename_component(target "${filename}" NAME)
        #     message(STATUS "Copying ${filename} to ${TARGET_NAME}/${target}")
        #     configure_file("${filename}" 
        #     "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}/${target}" COPYONLY)
        # endforeach (filename)
        # foreach (filename ${pyfiles_parser})
        #     get_filename_component(target "${filename}" NAME)
        #     message(STATUS "Copying ${filename} to ${TARGET_NAME}/${target}")
        #     configure_file("${filename}" 
        #     "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}/parser/${target}" COPYONLY)
        # endforeach (filename)

        # add_custom_target(PyPackageBuild
        # COMMAND "${PYTHON_EXECUTABLE}" -m pip wheel .
        # WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        # COMMENT "Building python wheel package"
        # )

        # execute_process(COMMAND "${PYTHON_EXECUTABLE}" getwheelname.py OUTPUT_VARIABLE PY_WHEELFILE
        # WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
        # message(STATUS "Detected Python wheel filename ${PY_WHEELFILE}")
        # set_target_properties(PyPackageBuild PROPERTIES Package_Name ${PY_WHEELFILE})
        # add_custom_target(PyPackageInstall
        #         COMMAND "${PYTHON_EXECUTABLE}" -m pip install --upgrade --force-reinstall --no-deps "${PY_WHEELFILE}"
        #         WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        #         COMMENT "Installing wheel file"
        #         )
        # add_dependencies(PyPackageInstall PyPackageBuild)
        # add_dependencies(PyPackageBuild ${TARGET_NAME})
        # if(DEFINED ENV{VIRTUAL_ENV} OR DEFINED ENV{CONDA_PREFIX})
        #     set(_pip_args)
        # else()
        #     set(_pip_args "--user")
        # endif()
        # execute_process(COMMAND ${Python_EXECUTABLE}
        #                 setup.py develop --user
        #                 WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/external/fffi)
    # endif()
endif()

# Add gyronimo
add_subdirectory(external/gyronimo)

# Add pybind11 include directories
execute_process(
    COMMAND "${PYTHON}" -c
    "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
    list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
find_package(pybind11 CONFIG REQUIRED)
# include_directories(${pybind11_INCLUDE_DIRS})

# Build NEAT
pybind11_add_module(${PROJECT_NAME}
    src/neatpp/fields_NEAT/equilibrium_stellna.cc
    src/neatpp/fields_NEAT/equilibrium_stellna_qs.cc
    src/neatpp/fields_NEAT/equilibrium_stellna_qs_partial.cc
    src/neatpp/metrics_NEAT/metric_stellna.cc
    src/neatpp/metrics_NEAT/metric_stellna_qs.cc
    src/neatpp/metrics_NEAT/metric_stellna_qs_partial.cc
    src/neatpp/neatpp.cpp
    src/neatpp/neatpp.hh
    # src/neatpp/NEAT_Mercier.cc
)

include_directories(${PROJECT_SOURCE_DIR}/src/neatpp/fields_NEAT)
include_directories(${PROJECT_SOURCE_DIR}/src/neatpp/metrics_NEAT)

if(PROFILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -lprofiler")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Create C++ library
target_link_libraries(${PROJECT_NAME} PRIVATE gyronimo)
target_link_libraries(${PROJECT_NAME} PRIVATE ${GSL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${Boost_LIBRARIES})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)

# Add the profiling executable
add_library(profiling_lib
    src/neatpp/fields_NEAT/equilibrium_stellna.cc
    src/neatpp/fields_NEAT/equilibrium_stellna_qs.cc
    src/neatpp/fields_NEAT/equilibrium_stellna_qs_partial.cc
    src/neatpp/metrics_NEAT/metric_stellna.cc
    src/neatpp/metrics_NEAT/metric_stellna_qs.cc
    src/neatpp/metrics_NEAT/metric_stellna_qs_partial.cc
    src/neatpp/neatpp.hh
)
target_link_libraries(profiling_lib PRIVATE gyronimo)
target_link_libraries(profiling_lib PRIVATE ${GSL_LIBRARIES})
target_compile_features(profiling_lib PRIVATE cxx_std_20)

add_executable(profiling src/neatpp/neatpp_profiling.cpp)
target_link_libraries(profiling PRIVATE profiling_lib)
target_link_libraries(profiling PRIVATE gyronimo)
target_link_libraries(profiling PRIVATE ${GSL_LIBRARIES})
target_compile_features(profiling PRIVATE cxx_std_20)
install(TARGETS profiling DESTINATION .)

## Attempt at making an executable from neatpp.cpp directly
# add_executable(profiling src/neatpp/neatpp.cpp)
# target_include_directories(profiling PRIVATE ${pybind11_INCLUDE_DIRS})
# target_link_libraries(profiling PRIVATE profiling_lib)
# target_link_libraries(profiling PRIVATE gyronimo)
# target_link_libraries(profiling PRIVATE ${GSL_LIBRARIES})
# target_link_libraries(profiling PRIVATE pybind11::module)
# target_compile_features(profiling PRIVATE cxx_std_20)
# install(TARGETS profiling DESTINATION .)

# Create python Wrapper for SIMPLE
if(INSTALL_SIMPLE_F90)
    # install(TARGETS simple DESTINATION neat)

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
    add_compile_options(-cpp)
    if (CMAKE_Fortran_COMPILER_ID MATCHES Intel)
        add_compile_options(-mkl -qopenmp -warn all,nounused -stand f2008)
        link_libraries("-mkl -qopenmp")
        if (CMAKE_BUILD_TYPE MATCHES Debug)
            add_compile_options(-O0 -g)
        elseif (CMAKE_BUILD_TYPE MATCHES Profile)
            add_compile_options(-O2 -g -shared-intel -debug inline-debug-info
                -D TBB_USE_THREADING_TOOLS -qopenmp-link dynamic -parallel-source-info=2)
        elseif (CMAKE_BUILD_TYPE MATCHES Fast)
            add_compile_options(-O3 -march=skylake -mtune=skylake)
        endif()
    else()
        add_compile_options(-std=f2008 -Wall -Wno-unused -Wno-unused-dummy-argument -fopenmp)
        link_libraries("-fopenmp")
        add_compile_options(-g)
        if (CMAKE_BUILD_TYPE MATCHES Debug)
            add_compile_options(-O0 -g -ggdb -C -p -fbacktrace 
                -ffpe-trap=invalid,zero,overflow -fbounds-check -fcheck=all,no-array-temps)
        elseif (CMAKE_BUILD_TYPE MATCHES Profile)
            add_compile_options(-O2 -g -shared-libgcc)
        elseif (CMAKE_BUILD_TYPE MATCHES Fast)
            add_compile_options(-O3 -march=ivybridge -mtune=ivybridge)
        endif()
    endif()
    # Ensure scikit-build modules
    execute_process(
        COMMAND "${Python_EXECUTABLE}"
        -c "import os, skbuild; print(os.path.dirname(skbuild.__file__))"
        OUTPUT_VARIABLE SKBLD_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(SKBLD_CMAKE_DIR "${SKBLD_DIR}/resources/cmake")
    list(APPEND CMAKE_MODULE_PATH ${SKBLD_CMAKE_DIR})
    # scikit-build style includes
    find_package(PythonExtensions REQUIRED) # for ${PYTHON_EXTENSION_MODULE_SUFFIX}
    find_package(NumPy REQUIRED) # for ${NumPy_INCLUDE_DIRS}
    find_package(F2PY REQUIRED) # for ${F2PY_INCLUDE_DIR}
    find_package(PythonLibs REQUIRED)
    # Project scope; consider using target_include_directories instead
    include_directories(
    BEFORE
    ${Python_INCLUDE_DIRS}
    ${Python_NumPy_INCLUDE_DIRS}
    ${F2PY_INCLUDE_DIR}
    )
    # Prepping the module
    set(f2py_module_name "neatf90")
    set(fortran_src_file
        # ${CMAKE_SOURCE_DIR}/external/simple/SRC/zzg.f90
        # ${CMAKE_SOURCE_DIR}/external/simple/SRC/sorting.f90
        # ${CMAKE_SOURCE_DIR}/external/simple/SRC/check_orbit_type.f90
        # external/simple/SRC/main.f90
        ${CMAKE_SOURCE_DIR}/src/neatf90/neatf90.f90
    )
    set(generated_module_file ${f2py_module_name}${PYTHON_EXTENSION_MODULE_SUFFIX})
    # Target for enforcing dependencies
    add_custom_target(${f2py_module_name} ALL
    DEPENDS "${fortran_src_file}"
    )
    # Custom command for generating .c
    add_custom_command(
    OUTPUT "${f2py_module_name}module.c" "${f2py_module_name}-f2pywrappers2.f90"
    COMMAND ${F2PY_EXECUTABLE}
        -m ${f2py_module_name}
        ${fortran_src_file}
        --lower
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${fortran_src_file}
    )
    # Create library
    add_library (simple_lib SHARED
        external/simple/SRC/canonical_coordinates_mod.f90
        external/simple/SRC/nctools_module.f90
        external/simple/SRC/odeint_rkf45.f90
        external/simple/SRC/contrib/minpack.f90
        external/simple/SRC/magfie.f90
        external/simple/SRC/boozer_converter.f90
        external/simple/SRC/binsrc.f90
        external/simple/SRC/plag_coeff.f90
        external/simple/SRC/chamb_m.f90
        external/simple/SRC/sub_alpha_lifetime_can.f90
        external/simple/SRC/vmecinm_m.f90
        external/simple/SRC/spline_vmec_data.f90
        external/simple/SRC/spl_three_to_five.f90
        external/simple/SRC/new_vmec_allocation_stuff.f90
        external/simple/SRC/get_canonical_coordinates.f90
        external/simple/SRC/testing.f90
        external/simple/SRC/field_can.f90
        external/simple/SRC/orbit_symplectic.f90
        external/simple/SRC/orbit_symplectic_quasi.f90
        external/simple/SRC/common.f90
        external/simple/SRC/simple.f90
        external/simple/SRC/bench.f90
        external/simple/SRC/parse_ants.f90
        external/simple/SRC/zzg.f90
        external/simple/SRC/sorting.f90
        external/simple/SRC/check_orbit_type.f90
    )
    add_library ( rkf45_lib SHARED
        external/simple/SRC/contrib/rkf45.f90
    )
    target_link_libraries(simple_lib rkf45_lib netcdf netcdff lapack blas)
    add_library(${generated_module_file} MODULE
                "${f2py_module_name}module.c"
                "${F2PY_INCLUDE_DIR}/fortranobject.c"
                "${fortran_src_file}"
                "${f2py_module_name}-f2pywrappers2.f90"
    )
    target_include_directories(${generated_module_file} PUBLIC
                ${F2PY_INCLUDE_DIR}
                ${F2PY_INCLUDE_DIRS}
                ${PYTHON_INCLUDE_DIRS}
                ${Python_NumPy_INCLUDE_DIRS}
    )
    target_link_libraries(${generated_module_file} PRIVATE simple_lib)
    target_link_libraries(${generated_module_file} PRIVATE ${PYTHON_LIBRARIES})
    set_target_properties(${generated_module_file} PROPERTIES SUFFIX "")
    set_target_properties(${generated_module_file} PROPERTIES PREFIX "")
    # Linker fixes
    if (UNIX)
    if (APPLE)
        set_target_properties(${generated_module_file} PROPERTIES
        LINK_FLAGS  '-Wl,-dylib,-undefined,dynamic_lookup')
    else()
        set_target_properties(${generated_module_file} PROPERTIES
    LINK_FLAGS  '-Wl,--allow-shlib-undefined')
    endif()
    endif()
    # Install
    install(TARGETS ${generated_module_file} DESTINATION ${CMAKE_SOURCE_DIR}/${f2py_module_name})

    # add_executable (simple.x
    # 	# ${CMAKE_SOURCE_DIR}/external/simple/SRC/zzg.f90
    # 	# ${CMAKE_SOURCE_DIR}/external/simple/SRC/sorting.f90
    # 	# ${CMAKE_SOURCE_DIR}/external/simple/SRC/check_orbit_type.f90
    # 	${CMAKE_SOURCE_DIR}/external/simple/SRC/main.f90
    # )
    # target_link_libraries(simple.x simple_lib)
    # install(TARGETS simple.x DESTINATION bin)
endif()