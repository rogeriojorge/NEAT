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
    docker run -v "$(pwd)/inputs.py:/usr/src/app/inputs.py" -v "$(pwd)/results:/usr/src/app/results" neat
    ```

3. Your results folder will be populated with NEAT's results

#### Optional
If you want to log into the container, first run
    ```bash
    docker run -dit neat
    ```
    then attach your terminal's input to the running container using the CONTAINER ID specified by the `docker ps` command
    ``` bash
    docker attach CONTAINER ID
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
g++ -O2 -Wall -shared -std=c++20 -undefined dynamic_lookup  NEAT.cc -o NEAT.so $(python3 -m pybind11 --includes) -I/opt/local/include -L/opt/local/lib -lgsl -L../build/lib -lgyronimo -I../build/include -Wl,-rpath ../build/lib
```

#### Example on Linux

```bash
g++-10 -std=c++2a -fPIC -shared NEAT.cc -o NEAT.so $(python3 -m pybind11 --includes) -L/usr/lib -lgsl -L../build/lib -lgyronimo -I../build/include/gyronimo -Wl,-rpath ../build/lib
```

# Profiling

## Install gperftools

The recommended way of installation in MacOS is through Macports (port install gperftools) or Homebrew (brew install gperftools).
In linux, the recommended way is through using apt (apt-get install google-perftools).

Alternatively, the gperftools/gperftools repository can be cloned to external/gperftools and installed in the build/ folder using the following commands:

```bash
cd external/gperftools
./autogen.sh
./configure --prefix=$PWD/../../build
make
make install
```

## Install yep

```bash
pip install yep
```

## Install gv

On MacOS **port install gv** or **brew install gv**
Open an instance of XQuartz

## Compile with gperftools linking

Example on MacOS

```bash
g++ -O2 -Wall -shared -std=c++20 -undefined dynamic_lookup  NEAT.cc -o NEAT.so $(python3 -m pybind11 --includes) -I/opt/local/include -L/opt/local/lib -lgsl-L../build/lib -lgyronimo -I../build/include -Wl,-rpath ../build/lib
```

## Run and Analyze

```bash
python -m yep -v -- main.py
pprof --gv main.py main.py.prof
```

