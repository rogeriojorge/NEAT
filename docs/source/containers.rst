.. _container_doc:

Containers
**********

.. _docker_doc:

Docker container
================

A Docker container for ``NEAT`` is available, allowing you to use
``NEAT`` without having to compile any code yourself.  The container
includes VMEC, pyQSC, pyQIC, SIMSOPT and SIMPLE.

.. warning::

   Docker is not generally allowed to run on computers at HPC centers due to security issues.
   For those wishing to run ``NEAT`` on MARCONI, NERSC, or other machines, please refer to udocker.

Requirements
^^^^^^^^^^^^
Docker needs to be installed before running the docker container. Docker
software can be obtained from `docker website <https://docs.docker.com/get-docker/>`_.
Check the `docker get started webpage <https://docs.docker.com/get-started/>`_ for installation instructions 
as well as for tutorials to get a feel for docker containers. On Linux and MacOS,
you may need to start the docker daemon before proceeding further.

.. warning::

   On Mac, the default 2 GB memory per container assigned by Docker Desktop may be too small.
   Increase the memory of the container to at least 3 GB to run ``NEAT`` much faster.

Install From Docker Hub
^^^^^^^^^^^^^^^^^^^^^^^
The easiest way to get ``NEAT`` docker image which comes with ``NEAT`` and all of its dependencies such as
SIMSOPT and VMEC pre-installed is to use Docker Hub. After 
`installing docker <https://docs.docker.com/get-started/>`_, you can run
the ``NEAT`` container directly from the ``NEAT`` docker image uploaded to
Docker Hub.

.. code-block::

   docker run -it --rm rjorge123/neat # Linux users, prefix the command with sudo

The above command should load the Ubuntu (Linux) terminal that comes with the NEAT
docker container. When you run it first time, the image is downloaded
automatically, so be patient.  You should now be able to import the module from
python::

  >>> python3
  >>> import neat

Note: to create files and store them locally, use the ``-v`` flag 
to mount the current directory

.. code-block:: 

    docker run -it --rm -v $PWD:/my_mount rjorge123/neat
    <container ###> cd /my_mount

Ways to use NEAT docker container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

IPython Shell
-------------

Easiest way is to start ipython shell and import the ``NEAT``
library. But this approach is only useful if a few commands need to be
executed or to load a python module and execute it.

.. code-block::

    docker run -it --rm rjorge123/neat ipython

To run the example in `examples/plot_single_orbit_qs.py` and check quantities
such as the canonical angular momentum and the total energy, simply run

.. code-block::

    %load examples/plot_single_orbit_qs.py
    g_orbit.p_phi
    g_orbit.total_energy

Jupyter notebook
----------------

The ``NEAT`` container comes with jupyter notebook preloaded. You can launch the jupyter from
the container using the command:

.. code-block::
   
    docker run -it --rm -v $PWD:/my_mount -p 8888:8888 rjorge123/neat
    <container ###> cd /my_mount
    <container ###> jupyter notebook --ip 0.0.0.0 --no-browser --allow-root 

Running the above command, a link will be printed that you can use to
open the jupyter console in a web browser. The link typically starts
with ``http://127.0.0.1:8888/?token=``. (Several links are printed,
but only the last one listed will work for browsers running outside
the container.) Copy the full link and paste it into any browser in
your computer. You now see the file tree. On the upper right corner,
you can create a new notebook to run python directly there.
To run the example in `examples/plot_single_orbit_qs.py`, you can simply type

  %load examples/plot_single_orbit_qs.py

and hit shift enter twice. The example will now be shown.


Persistent containers
^^^^^^^^^^^^^^^^^^^^^

Using the intructions above will create a fresh container each time and delete the container after exiting.
If you would like to create a persistent container (e.g. because you are installing additional pip packages inside) that you can reuse at any time,
you can do so by removing the ``--rm`` command and specifying a container name via ``--name=``

.. code-block::

    docker run --name=mycontainer -it -v $PWD:/my_mount rjorge123/neat
    <container ###> cd /my_mount
    <container ###> python3 <driver_script>

And to restart and rejoin the container:

.. code-block::

    docker start mycontainer
    docker exec -it mycontainer /bin/bash

