C++ backend
***********

``NEAT`` uses C++ for performance critical functions such as the guiding center integration using the ``gyronimo`` library.
This section is aimed at advanced developers of ``NEAT`` to give an overview over the interface between C++ and Python and to help avoid common pitfalls. For most users of ``NEAT`` this is not relevant.

The C++ code can be found in the folder ``src/neatpp``.


pybind11
^^^^^^^^

To expose the C++ code to python, we use the 
`pybind11 <https://github.com/pybind/pybind11>`_ library.
The interfacing happens in the ``neatpp.cpp`` and ``neatpp.hh`` files.


Lifetime of objects:
memory management in hybrid Python/C++ codes can be difficult. To guarantee that objects managed by C++ are not deleted even though they are still used on the Python side, we make sure that we hold a reference to them in the Python code.


OpenMP
^^^^^^
Some of the code is parallelized using OpenMP. OpenMP can be turned of by setting
``export OMP_NUM_THREADS=1``
before running a ``NEAT`` script. This is recommended when debugging bugs that are assumed to be in the C++ code.


CMake
^^^^^

When editing the C++ code, it may be useful to use ``CMake`` and ``make`` directly to only recompile those parts of the code that changed. This can be achieved as below

.. code-block::

    git clone --recursive git@github.com:rogeriojorge/neat.git
    cd neat
    pip3 install -e .
    mkdir cmake-build
    cd cmake-build
    cmake ..
    make -j
    cd ../src
    rm neatpp.cpython-38-x86_64-linux-gnu.so
    ln -s ../cmake-build/neatpp.cpython-38-x86_64-linux-gnu.so .

You may have to adjust the last two lines to match your local system.
From then on, you can always just call ``make -j`` inside the ``cmake-build`` directory to recompile the C++ part of the code.
