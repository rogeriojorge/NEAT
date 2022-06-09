Overview
========

Ways to use NEAT
-------------------

NEAT is a collection of classes and functions that can be used in
several ways.  One application is to solve optimization problems
involving stellarators, similar to SIMSOPT.  You could also define an
objective function using SIMSOPT, but use an optimization library
outside SIMSOPT to solve it.  Or, you could use the SIMSOPT
optimization infrastructure to optimize your own objects, which may or
may not have any connection to stellarators.  Alternatively, you can
use the stellarator-related objects in a script for some purpose other
than optimization, such as plotting how some code output varies as an
input parameter changes, or evaluating the finite-difference gradient
of some code outputs.  Or, you can manipulate the objects
interactively, at the python command line or in a Jupyter notebook.


Input files
-----------

NEAT does not use input data files to define optimization problems,
in contrast to ``STELLOPT``. Rather, problems are specified using a
python driver script, in which objects are defined and
configured. However, objects related to specific physics codes may use
their own input files.


Modules
-------

Classes and functions in NEAT are organized into several modules:

- :obj:`neat.constants` contains several physical constants.
- :obj:`neat.fields` contains the classes for the stellarator magnetic fields, including the near-axis field.
- :obj:`neat.objectives` contains the objective functions used for optimizing stellarators.
- :obj:`neat.plotting` contains the plotting and animation routines.
- :obj:`neat.tracing` contains the tracing classes that treats the input and output to the ``neatpp`` package.
