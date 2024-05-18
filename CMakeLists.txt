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
find_package(Python 3 COMPONENTS Interpreter Development NumPy)
# Use relative paths in MacOS
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_LIBDIR lib)
add_compile_options(-Wfatal-errors -Wpedantic)
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

# Use netcdf if libraries are found. The values of NETCDF_LIB_PATH and NETCDF_INC_PATH should be set from the command line or Cmake GUI. Convert path from native format to cmake internal format.
message(STATUS "NETCDF_INC_PATH is ${NETCDF_INC_PATH}")
if (NETCDF_INC_PATH)
    file (TO_NATIVE_PATH ${NETCDF_INC_PATH} NETCDF_INC_PATH)
	find_file (NETCDF_I netcdf.inc ${NETCDF_INC_PATH} NO_DEFAULT_PATH)
    find_file (NETCDF_M netcdf.mod ${NETCDF_INC_PATH} NO_DEFAULT_PATH)
else ()
    get_directory_property(INC_DIRS INCLUDE_DIRECTORIES)
    message(STATUS "INC_DIRS  are ${INC_DIRS}")
	find_file (NETCDF_I netcdf.inc ${INC_DIRS} NO_DEFAULT_PATH)
    find_file (NETCDF_M netcdf.mod ${INC_DIRS} NO_DEFAULT_PATH)
endif ()
message(STATUS "NETCDF_I value is ${NETCDF_I}")
message(STATUS "NETCDF_M value is ${NETCDF_M}")

if (NETCDF_LIB_PATH)
    file (TO_NATIVE_PATH ${NETCDF_LIB_PATH} NETCDF_LIB_PATH)
    find_library (NETCDF_F NAMES netcdff ${NETCDF_F_LIB_NAME} HINTS ${NETCDF_LIB_PATH} ENV LD_LIBRARY_PATH)
    find_library (NETCDF_C NAMES netcdf ${NETCDF_C_LIB_NAME} HINTS ${NETCDF_LIB_PATH} ENV LD_LIBRARY_PATH)
else()
    get_directory_property(LINK_DIRS LINK_DIRECTORIES)
    find_library (NETCDF_F NAMES netcdff ${NETCDF_F_LIB_NAME} HINTS ${LINK_DIRS} ENV LD_LIBRARY_PATH)
    find_library (NETCDF_C NAMES netcdf ${NETCDF_C_LIB_NAME} HINTS ${LINK_DIRS} ENV LD_LIBRARY_PATH)
endif ()
message(STATUS "NETCDF_F value is ${NETCDF_F}")
message(STATUS "NETCDF_C value is ${NETCDF_C}")
if (NETCDF_C_LIB_PATH)
    find_library (NETCDF_C NAMES netcdf ${NETCDF_C_LIB_NAME} HINTS ${NETCDF_C_LIB_PATH} ENV LD_LIBRARY_PATH)
endif ()

if (NETCDF_I AND (NETCDF_F OR NETCDF_C))
	message (STATUS "Using netcdf")
	add_definitions (-DNETCDF)
	include_directories (${NETCDF_INC_PATH})

	if (NETCDF_F)
		list (APPEND NETCDF_LIBRARIES ${NETCDF_F})
	endif ()
	if (NETCDF_C)
		list (APPEND NETCDF_LIBRARIES ${NETCDF_C})
	endif ()
else ()
	message (FATAL_ERROR "Failed to find the required netcdf libraries. These must be found by setting the NETCDF_INC_PATH and NETCDF_LIB_PATH variables.")
endif ()

include_directories(${NETCDF_INC_PATH})
link_directories(${NETCDF_LIB_PATH})

# Allow users to define include and link directories
set(USER_INC_DIR "" CACHE FILEPATH "Set paths to include folders")
set(USER_LIB_DIR "" CACHE FILEPATH "Set paths to link folders")

include_directories(${USER_INC_DIR})
link_directories(${USER_LIB_DIR})

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
    # include_directories("${CMAKE_CURRENT_BINARY_DIR}/external/simple")
    # add_subdirectory(src/neatf90)
endif()

# Add gyronimo with VMEC support
set(BUILD_VMEC_SUPPORT "ON")
add_subdirectory(external/gyronimo)

# Add pybind11 include directories
execute_process(
    COMMAND "${PYTHON}" -c
    "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
    list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
include_directories(external/gyronimo/misc/apps/include)
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
    src/neatpp/parsers_NEAT/parser_boozxform.cc
    src/neatpp/stellna.hh
    src/neatpp/vmectrace.hh
    src/neatpp/vmec_loss.hh
    src/neatpp/booztrace.hh
    src/neatpp/neatpp.cpp
    # src/neatpp/NEAT_Mercier.cc
    src/neatpp/fields_NEAT/equilibrium_boozxform.cc
    src/neatpp/metrics_NEAT/metric_boozxform.cc
)

include_directories(${PROJECT_SOURCE_DIR}/src/neatpp/fields_NEAT)
include_directories(${PROJECT_SOURCE_DIR}/src/neatpp/metrics_NEAT)
include_directories(${PROJECT_SOURCE_DIR}/src/neatpp/parsers_NEAT)

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

if(PROFILING)
    # Add the profiling executable
    add_library(profiling_lib
        src/neatpp/fields_NEAT/equilibrium_stellna.cc
        src/neatpp/fields_NEAT/equilibrium_stellna_qs.cc
        src/neatpp/fields_NEAT/equilibrium_stellna_qs_partial.cc
        src/neatpp/metrics_NEAT/metric_stellna.cc
        src/neatpp/metrics_NEAT/metric_stellna_qs.cc
        src/neatpp/metrics_NEAT/metric_stellna_qs_partial.cc
        # src/neatpp/fields_NEAT/equilibrium_vmec_optimized.cc
        # src/neatpp/metrics_NEAT/metric_vmec_optimized.cc
        src/neatpp/stellna.hh
        src/neatpp/vmectrace.hh
        src/neatpp/vmec_loss.hh
        src/neatpp/booztrace.hh
        src/neatpp/fields_NEAT/equilibrium_boozxform.cc
        src/neatpp/metrics_NEAT/metric_boozxform.cc
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
endif()

## Attempt at making an executable from neatpp.cpp directly
# add_executable(profiling src/neatpp/neatpp.cpp)
# target_include_directories(profiling PRIVATE ${pybind11_INCLUDE_DIRS})
# target_link_libraries(profiling PRIVATE profiling_lib)
# target_link_libraries(profiling PRIVATE gyronimo)
# target_link_libraries(profiling PRIVATE ${GSL_LIBRARIES})
# target_link_libraries(profiling PRIVATE pybind11::module)
# target_compile_features(profiling PRIVATE cxx_std_20)
# install(TARGETS profiling DESTINATION .)