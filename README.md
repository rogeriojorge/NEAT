## NEAT
NEar-Axis opTimisation

## Install NEAT
gyronimo and pybind11 added to NEAT using git submodules
To download gyronimo and pybind11 when cloning NEAT clone using the following command:
git clone --recurse-submodules https://github.com/rogeriojorge/NEAT.git
or, alternatively, after cloning NEAT, run:
git submodule init
git submodule update

## NEAT Docker Container
This document explains how to build the docker container for simsopt.
This process yields an image with roughly 900 MB and may take a minute to load.

How to build the container:
0. Install docker
1. Build the docker image by running the `docker build` command in the repo root directory:
   ```bash
   docker build -t neat -f Dockerfile.NEAT .
   ```
2. Run the docker image using the `docker run` command:
    ``` bash
    docker run -dit neat
    ```
3. Attach your terminal's input to the running container using the CONTAINER ID specified by the `docker ps` command
    ``` bash
    docker attach CONTAINER ID
    ```