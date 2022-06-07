===================
NEAT Documentation
===================

``NEAT`` is a hybrid python/C++ framework for optimizing
`stellarators <https://en.wikipedia.org/wiki/Stellarator>`_
for good fast-particle confinement using the near-axis expansion formalism.
You can read more about these concepts in the overview section :doc:`here <overview>`.
The high-level routines are written in python, with calls to C++ where needed for performance.
The main components of ``NEAT`` are:

* Interface with physics codes such as
  `gyronimo <https://github.com/prodrigs/gyronimo>`_,
  `pyQSC <https://github.com/landreman/pyQSC>`_,
  `pyQIC <https://github.com/rogeriojorge/pyQIC>`_,
  `SIMSOPT <https://github.com/hiddenSymmetries/simsopt>`_ and
  `SIMPLE <https://github.com/itpplasma/SIMPLE>`_.
* Simple to use classes to trace particles in stellarator magnetic fields.
* Tools for defining objective functions and parameter spaces for optimization.

The design of ``NEAT`` is guided by several principles:

- Thorough unit testing, regression testing, and continuous integration.
- Extensibility: It should be possible to add new codes and terms to
  the objective function without editing modules that already work,
  i.e. the `open-closed principle
  <https://en.wikipedia.org/wiki/Open%E2%80%93closed_principle>`_.
  This is because any edits to working code can potentially introduce bugs.
- Modularity: Physics modules that are not needed for your
  problem do not need to be installed. For instance, to
  just trace particles you don't need the ``simsopt`` or ``SIMPLE`` packages.
- Flexibility: The components used to define a particles, orbits or objective functions
  can be re-used for applications other than standard optimization

We gratefully acknowledge funding from the `EUROfusion
<https://www.euro-fusion.org/news/2021/march/eurofusion-awards-16-enabling-research-projects/>`_
and from `Fundação para a Ciência e Tecnologia, Portugal (FCT)
<https://www.fct.pt/noticias/index.phtml.pt?id=734&/2021/11/Resultados_do_4%C2%BA_Concurso_de_Est%C3%ADmulo_do_Emprego_Cient%C3%ADfico_%E2%80%93_Individual>`_.

``NEAT`` is fully open-source, and anyone is welcome to
make suggestions, contribute, and use.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   installation
   containers
   testing
   source
   publications
   contributing
   cpp

.. toctree::
   :maxdepth: 3
   :caption: Tutorials

   example_plot
   example_loss_fraction
   example_optimize

.. toctree::
   :maxdepth: 3
   :caption: API

   neat

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`