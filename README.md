## NEAT
NEar-Axis opTimisation

# Requirements
To run NEAT, you'll need the following libraries

* gsl
* boost
* gcc10

and the python packages specified in requirements.txt .

# Download NEAT
gyronimo and pybind11 added to NEAT using git submodules
To download gyronimo and pybind11 when cloning NEAT clone using the following command:
```bash
git clone --recurse-submodules https://github.com/rogeriojorge/NEAT.git
```
or, alternatively, after cloning NEAT, run:
```bash
git submodule init
git submodule update
```

# Usage

## Recommended

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
