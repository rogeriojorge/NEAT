Installation
============

This page provides general information on installation.
.. Detailed installation instructions for some specific systems will be made available
.. on `the wiki <https://github.com/rogeriojorge/neat/wiki>`_.

Requirements
^^^^^^^^^^^^

``NEAT`` is a python package focused on stellarator optimization
and requires python version 3.7 or higher.  ``NEAT`` also requires
some mandatory python packages, listed in
`requirements.txt <https://github.com/rogeriojorge/neat/blob/master/requirements.txt>`_
and in the ``[options]`` section of
`setup.cfg <https://github.com/rogeriojorge/neat/blob/master/setup.cfg>`_.
These packages are all installed automatically when you install using
``pip`` or another python package manager such as ``conda``, as
discussed below.  If you prefer to install via ``python setup.py
install`` or ``python setup.py develop``, you will need to install
these python packages manually using ``pip`` or ``conda``, e.g.
with ``pip install -r requirements.txt``.


Virtual Environments
^^^^^^^^^^^^^^^^^^^^


This is an optional step, but users are strongly encouraged to use a python virtual environment
to install NEAT. There are two popular ways to create a python virtual environment using 
either ``venv`` module supplied with python or the conda virtual environment.

venv
----

A python virtual envionment can be created with venv using

.. code-block::

    python3 -m venv <path/to/new/virtual/environment>

Activate the newly created virtual environmnet (for bash shell)

.. code-block::
   
    . <path/to/new/virtual/environment>/bin/activate

If you are on a different shell, use the ``activate`` file with an appropriate extension reflecting the shell type.
For more information, please refer to `venv official documentation <https://https://docs.python.org/3/library/venv.html>`_.

conda
-----
Install either `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or `anaconda <https://www.anaconda.com/>`_.
If you are on a HPC system, anaconda is either available by default or via a module.

A conda python virtual environment can be created by running

.. code-block::

    conda create -n <your_virtual_env_name> python=3.8

For the new virtual environment, python version 3.8 was chosen in the above command, but you are free to choose any version you want. 
The newly created virtual environment can be activated with a simple command

.. code-block::

    conda activate <your_virtual_env_name>

After activating the conda virtual environment, the name of the environment should appear in the shell prompt.

Installation methods
^^^^^^^^^^^^^^^^^^^^

PyPi
----

This works for both venv and conda virtual environments.

.. code-block::

    pip install neatstel

Running the above command will install neat and all of its mandatory dependencies. 
    
On some systems, you may not have permission to install packages to
the default location. In this case, add the ``--user`` flag to ``pip``
so the package can be installed for your user only::

    pip install --user neatstel


From source
-----------

This approach works for both venv and conda virtual environments.
First, install ``git`` if not already installed. Then clone the repository using

.. code-block::

    git clone https://github.com/rogeriojorge/neat.git

Then install the package to your local python environment with

.. code-block::

    cd NEAT
    pip install -e .

The ``-e`` flag makes the installation "editable", meaning that the
installed package is a pointer to your local repository rather than
being a copy of the source files at the time of installation. Hence,
edits to code in your local repository are immediately reflected in
the package you can import.

Again, if you do not have permission to install python packages to the
default location, add the ``--user`` flag to ``pip`` so the package
can be installed for your user only::

    pip install --user -e .
    
.. warning::
    Installation from local source creates a directory called **build**. If you are reinstalling NEAT from source after updating the code by making local changes or by git pull, remove the directory **build** before reinstalling.


Docker container
----------------

A docker image with NEAT along with its dependencies, VMEC, SIMSOPT,
and SIMPLE pre-installed is available from docker hub. This
container allows you to use NEAT without having to compile any code
yourself.  After `installing docker
<https://docs.docker.com/get-docker/>`_, you can run the NEAT
container directly from the docker image uploaded to Docker Hub.

.. code-block::

   docker run -it --rm rjorge123/neat python3

The above command should load the python shell that comes with the
NEAT docker container. When you run it first time, the image is
downloaded automatically, so be patient. More information about using
simsopt with Docker can be found :doc:`here <containers>`.

Post-Installation
^^^^^^^^^^^^^^^^^

If the installation is successful, ``NEAT`` will be added to your
python environment. You should now be able to import the module from
python::

  >>> import neat
  >>> import neatpp

