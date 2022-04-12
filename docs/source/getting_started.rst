NEAT could be run either directly by installing the requirements pyQSC, gyronimo and SIMSOPT, and then compiling the [NEAT.cc](src/NEAT.cc) file in the *[src](src/)* folder, or using the provided Docker image. The usage of the Docker image is recommended.

Download
^^^^^^^^

To download clone NEAT including its submodules, use the following command:

```bash
git clone --recurse-submodules https://github.com/rogeriojorge/NEAT.git
```
or, alternatively, after downloading this repository, in the root folder, run:

```bash
git submodule init
git submodule update
```

Installation
^^^^^^^^^^^^

CMake

On NEAT's root directory run

```
python setup.py build
python setup.py install --user
```

To clean the build folders and all folders not being tracked by GIT, run

```
git clean -d -f -x
```

Docker
^^^^^^

This section explains how to build the docker container for NEAT. It can be used to compile gyronimo, install pyQSC, simsopt and compile NEAT in a docker image directly.

1. Install docker

2. Build the docker image by running the `docker build` command in the repo root directory:
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
