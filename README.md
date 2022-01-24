# **NEAT**
NEar-Axis opTimisation

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

## Docker

This section explains how to build the docker container for NEAT. It can be used to compile gyronimo, install pyQSC, simsopt and compile NEAT in a docker image directly.

0. Install docker

1. Build the docker image by running the `docker build` command in the repo root directory:
   ```bash
   docker build -t neat -f docker/Dockerfile.NEAT .
   ```
This process yields an image with roughly 2 GB and may take minute to build.

2. Run the docker image using the `docker run` command including your inputs file and results folder:
    ``` bash
    docker run -v "$(pwd)/inputs.py:/home/neat/src/inputs.py" -v "$(pwd)/results:/home/neat/results" neat
    ```

3. Your results folder will be populated with NEAT's results

#### Optional
If you want to run NEAT and continue working in the container, instead run the docker image using the flag **-it** and end with **/bin*bash**
    ```bash
    docker run -v "$(pwd)/inputs.py:/home/neat/src/inputs.py" -v "$(pwd)/results:/home/neat/results" -it neat /bin/bash
    ```

## Development

### Requirements
To run NEAT, you'll need the following libraries

* gsl
* boost
* gcc10

and the python packages specified in [requirements.txt](requirements.txt) .

### Install gyronimo
In NEAT's root folder, run

```bash
cd external/gyronimo
mkdir build
cd build
CXX=g++ cmake -DCMAKE_INSTALL_PREFIX=../../../build ../
cmake --build . --target install
```

If you want to build documentation with doxygen, run

```bash
cmake --build . --target doc
```

### Compile NEAT

Compilation is done in the src/ folder of the repo.

#### Example on MacOS

```bash
g++ -O2 -Wall -shared -std=c++20 -undefined dynamic_lookup  NEAT.cc -o NEAT.so $(python3 -m pybind11 --includes) -I/opt/local/include -L/opt/local/lib -lgsl -L$(pwd)/../build/lib -lgyronimo -I$(pwd)/../build/include -Wl,-rpath $(pwd)/../build/lib
```

#### Example on Linux

```bash
g++-10 -std=c++2a -fPIC -shared NEAT.cc -o NEAT.so $(python3 -m pybind11 --includes) -L/usr/lib -lgsl -L$(pwd)/../build/lib -lgyronimo -I$(pwd)/../build/include  -Wl,-rpath $(pwd)/../build/lib
```

# Profiling

## test_openmp_stellna

An simple openmp version of for OpenMP profiling purposes is in tests/test_openmp_stellna. After compilation

```bash
g++ -std=c++20 -fopenmp openmp_stellna.cc -o openmp_stellna -I/opt/local/include -L/opt/local/lib -lgsl -L$(pwd)/../build/lib -lgyronimo -I$(pwd)/../build/include  -Wl,-rpath $(pwd)/../build/lib
```

it creates an executable that can be analyzed using a profiling tool such as Instruments in MacOS, specifically the Time Profiler. The number of threads can be changed using the command

```bash
export OMP_NUM_THREADS=[number of threads]
```