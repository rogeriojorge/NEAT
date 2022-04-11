===================
NEAT Documentation
===================

![GitHub](https://img.shields.io/github/license/rogeriojorge/neat)
[![Documentation Status](https://readthedocs.org/projects/neat-docs/badge/?version=latest)](https://neat-docs.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/rogeriojorge/NEAT/branch/main/graph/badge.svg?token=8515A2RQL3)](https://codecov.io/gh/rogeriojorge/NEAT)

NEAT is a python framework that is intended to find optimized stellarator configurations for fast particle confinement using the near-axis expansion formalism.
The magnetic field is calculated using the code [pyQSC](https://github.com/landreman/pyQSC/), the particle orbits are traced using the code [gyronimo](https://github.com/prodrigs/gyronimo) (included as a submodule) and the optimization is done using the code [simsopt](https://github.com/hiddenSymmetries/).

``pyQSC`` is fully open-source, and anyone is welcome to
make suggestions, contribute, and use.

.. toctree::
   :maxdepth: 3

   getting_started
   usage
   source