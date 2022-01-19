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
mkdir build
cd build
CXX=g++ cmake ../external/gyronimo
make
```
If you want to build documentation, run
```bash
make doc
```

### Compile NEAT

Example on MacOS

```bash
g++ -O2 -Wall -shared -std=c++20 -undefined dynamic_lookup  NEAT.cc -o NEAT.so $(python3 -m pybind11 --includes) -I/opt/local/include -L/opt/local/lib -lgsl -L../build -lgyronimo -I../external/gyronimo/ -Wl,-rpath ../build
```

Example on Linux

```bash
RUN g++-10 -std=c++2a -fPIC -shared NEAT.cc -o NEAT.so $(python -m pybind11 --includes) -L/usr/lib -lgsl -L../build -lgyronimo -I../external/gyronimo -Wl,-rpath ../build
```
