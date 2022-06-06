
# **NEAT**
**NEar-Axis opTimisation**

![GitHub](https://img.shields.io/github/license/rogeriojorge/neat)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/rogeriojorge/NEAT/CI)
[![Documentation Status](https://readthedocs.org/projects/neat-docs/badge/?version=latest)](https://neat-docs.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/rogeriojorge/NEAT/branch/main/graph/badge.svg?token=8515A2RQL3)](https://codecov.io/gh/rogeriojorge/NEAT)

NEAT is a python framework that is intended to find optimized stellarator configurations for fast particle confinement using the near-axis expansion formalism.
The magnetic field is calculated using the code [pyQSC](https://github.com/landreman/pyQSC/), the particle orbits are traced using the code [gyronimo](https://github.com/prodrigs/gyronimo) (included as a submodule) and the optimization is done using the code [simsopt](https://github.com/hiddenSymmetries/).

To download clone NEAT including its submodules, use the following command:

```bash
git clone --recurse-submodules https://github.com/rogeriojorge/NEAT.git
```
or, alternatively, after downloading this repository, in the root folder, run:

```bash
git submodule init
git submodule update
```

# Usage

NEAT could be run either directly by installing the requirements pyQSC, gyronimo and SIMSOPT, and then compiling the [NEAT.cc](src/NEAT.cc) file in the *[src](src/)* folder, or using the provided Docker image. The usage of the Docker image is recommended.

# Installation

## CMake


Make sure that you have installed all of the python packages listed in the file [requirements.txt](requirements.txt). A simple way of doing so is by running

```
pip install -r requirements.txt
```

On NEAT's root directory run

```
python setup.py build
python setup.py install --user
```

Done! Now try running an example.

## Docker

This section explains how to build the docker container for NEAT. It can be used to compile gyronimo, install pyQSC, simsopt and compile NEAT in a docker image directly.

### Using Docker Hub

The easiest way to get simsopt docker image which comes with NEAT and all of its dependencies such as gyronimo and VMEC pre-installed is to use Docker Hub. After installing docker, you can run the simsopt container directly from the simsopt docker image uploaded to Docker Hub.

```
docker run -it --rm rjorge123/neat # Linux users, prefix the command with sudo
```

The above command should load the terminal that comes with the NEAT docker container. When you run it first time, the image is downloaded automatically, so be patient. You should now be able to import the module from python:

```
python3
import neat
```

### Build locally

To build the image locally, instead of downloading from DockerHub, you can use the commands below:


1. Build the docker image by running the `docker build` command in the repo root directory:
   ```bash
   docker build -t neat -f docker/Dockerfile.NEAT .
   ```
This process yields an image with roughly 2 GB and may take minute to build.

2. Run the docker image using the `docker run` command including your results folder:
    ``` bash
    docker run -v "$(pwd)/results:/home/results" neat
    ```

3. Your results folder will be populated with NEAT's results

4. In case the input parameters are changed, there is no need to rebuild the image, just include your inputs file after the docker run command
    ``` bash
    docker run -v "$(pwd)/inputs.py:/home/neat/src/inputs.py" -v "$(pwd)/results:/home/neat/results" neat
    ```

#### Optional
If you want to run NEAT and continue working in the container, instead run the docker image using the flag **-it** and end with **/bin*bash**
    ```bash
    docker run -it --entrypoint /bin/bash neat
    ```

## Development

### Requirements
To run NEAT, you'll need the following libraries

* gsl
* boost
* gcc10

and the python packages specified in [requirements.txt](requirements.txt) .

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

    ```https://github.com/gperftools/gperftools```

On MacOs, it can be installed via Macports or Homebrew.
On Ubuntu, it can be install via ```sudo apt-get install google-perftools```.

For it to profile the code, the flag `PROFILING` should be `ON` in the `cmake_config_file.json` file.
After compiling NEAT, it will create an executable called `profiling` in the temporary build directory.
To profile the code using the `gperftools`, you can run

    ```CPUPROFILE=profile.out build/path_to_profiling```

where the output file for the profiling was named `profile.out`.

The results can be ploted using the following command

    ```pprof --gv build/path_to_profiling profile.out```

On MacOs, to show the plot, one needs to install `gprof2dot`, `graphivz` and `gv`. On macports, for example, this can be done using

    ```sudo port install py310-gprof2dot graphivz gv```

where the python version 3.10 was specified.

If, instead of plotting, you would like text results, you can run

    ```prof build/path_to_profiling profile.out```

# FAQ

## pybind11 not found by cmake

Please use the following command to install ```pybind11[global]``` instead of ```pybind11```

```
pip install "pybind11[global]"
```

## How to clean all folders created during installation/execution

To clean the build folders and all folders not being tracked by GIT, run

```
git clean -d -f -x
```