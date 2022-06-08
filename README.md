
# **NEAT**
**NEar-Axis opTimisation**

![GitHub](https://img.shields.io/github/license/rogeriojorge/neat)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/rogeriojorge/NEAT/CI)
[![Documentation Status](https://readthedocs.org/projects/neat-docs/badge/?version=latest)](https://neat-docs.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/rogeriojorge/NEAT/branch/main/graph/badge.svg?token=8515A2RQL3)](https://codecov.io/gh/rogeriojorge/NEAT)

NEAT is a python framework that is intended to find optimized stellarator configurations for fast particle confinement using the near-axis expansion formalism.
The magnetic field is calculated using the codes [pyQSC](https://github.com/landreman/pyQSC/) and [pyQIC](https://github.com/rogeriojorge/pyQIC/), the particle orbits are traced using the code [gyronimo](https://github.com/prodrigs/gyronimo) (included as a submodule) and the optimization is done using the code [simsopt](https://github.com/hiddenSymmetries/). The benchmarks are made with the [SIMPLE](https://github.com/itpplasma/SIMPLE) and [BEAMS3D](https://github.com/PrincetonUniversity/STELLOPT/tree/develop/BEAMS3D) codes (under construction).

We show here the standard way to download and install NEAT. For more information, please visit the [documentation](http://neat-docs.readthedocs.io/) present in http://neat-docs.readthedocs.io/.

# Installation

To install NEAT, you'll need the following libraries

* gsl
* boost
* gcc10

and the python packages specified in [requirements.txt](requirements.txt).
Note that [pyQSC](https://github.com/landreman/pyQSC/) and [pyQIC](https://github.com/rogeriojorge/pyQIC/) should be downloaded and installed locally.

## PyPI

The simplest installation of NEAT is by running the command

    pip install neatstel

However, it doesn't work on every system and the code hosted in PyPI might be outdated.

## From source

To download, clone NEAT using the following command:

    git clone https://github.com/rogeriojorge/NEAT.git


The python packages necessary to run NEAT are listed in the file [requirements.txt](requirements.txt).
A simple way of installing them is by running


    pip install -r requirements.txt


Then, to install NEAT, on its root directory run


    pip install -e .


If you do not have permission to install python packages to the
default location, add the ``--user`` flag to ``pip`` so the package
can be installed for your user only::


    pip install --user -e .


To debug any possible problems that might arise, you may also try to install
using the provided ``setup.py`` file


    python setup.py install --user


Done! Now try running an example.

Note: the python package is called `neatstel`.

## Docker

This section explains how to build the docker container for NEAT. It can be used to compile gyronimo, install pyQSC, simsopt and compile NEAT in a docker image directly.

### Using Docker Hub

The easiest way to get simsopt docker image which comes with NEAT and all of its dependencies such as gyronimo and VMEC pre-installed is to use Docker Hub. After installing docker, you can run the simsopt container directly from the simsopt docker image uploaded to Docker Hub.


    docker run -it --rm rjorge123/neat


The above command should load the terminal that comes with the NEAT docker container. When you run it first time, the image is downloaded automatically, so be patient. You should now be able to import the module from python:


    python3
    import neat


### Build locally

To build the image locally, instead of downloading from DockerHub, you can use the commands below:


1. Build the docker image by running the `docker build` command in the repo root directory:

   docker build -t neat -f docker/Dockerfile.NEAT .

This process yields an image with roughly 2 GB and may take minute to build.

2. Run the docker image using the `docker run` command including your results folder:

    docker run -it neat


3. Done! You are now in an environment with NEAT installed. You can open python and run the examples.

# Usage

For common uses of NEAT, please check the `examples` folder.
It has the three main current uses of NEAT:
- Trace a single particle orbit (`examples/plot_single_orbit_qs.py`)
- Trace an ensemble of particles (`examples/calculate_loss_fraction_qs.py`)
- Optimize a stellarator magnetic field (`examples/optimize_loss_fraction_qs.py`)

# Normalizations

All units are in SI, except:
- The mass of the particle, which is in units of the mass of the proton
- The charge of the particle is normalized to the charge of the proton
- The energy is in eV

Lambda = mu (SI) / Energy (SI) * B (reference SI)

# Profiling

## Profiling Python code

Use the line_profiler python extension.

```pip install line_profiler```

Example [here](https://stackoverflow.com/questions/22328183/python-line-profiler-code-example/43376466#43376466)

## Profiling the C++ extension

There is a C++ script in the `src/neatpp` directory called `neatpp_profiling.cpp` that has the
sole purpose of helping find bottlenecks in the C++ implementation. We show here an example of
how to profile the code using the tool `gperftools`.

    https://github.com/gperftools/gperftools

On MacOs, it can be installed via Macports or Homebrew.
On Ubuntu, it can be install via ```sudo apt-get install google-perftools```.

For it to profile the code, the flag `PROFILING` should be `ON` in the `cmake_config_file.json` file.
After compiling NEAT, it will create an executable called `profiling` in the temporary build directory.
To profile the code using the `gperftools`, you can run

    CPUPROFILE=profile.out build/path_to_profiling

where the output file for the profiling was named `profile.out`.

The results can be ploted using the following command

    pprof --gv build/path_to_profiling profile.out

On MacOs, to show the plot, one needs to install `gprof2dot`, `graphivz` and `gv`. On macports, for example, this can be done using

    sudo port install py310-gprof2dot graphivz gv

where the python version 3.10 was specified.

If, instead of plotting, you would like text results, you can run

    prof build/path_to_profiling profile.out

# FAQ

## pybind11 not found by cmake

Please use the following command to install ```pybind11[global]``` instead of ```pybind11```


    pip install "pybind11[global]"


## How to clean all folders created during installation/execution

To clean the build folders and all folders not being tracked by GIT, run

    git clean -d -f -x
