.. GDA Public Toolbox documentation master file, created by
   sphinx-quickstart on Tue May 17 09:10:25 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GDA Toolbox (public version)
============================

Public Repository at https://github.com/geomdata/gda-public/

Documentation at https://geomdata.github.io/gda-public/

This package provides easy-to-use tools for manipulating time-series signals
(see the `timeseries.Signal` class), point clouds (see the
`multidim.PointCloud` class) and simplicial complexes (see the
`multidim.SimplicialComplex` class).  The goal is to provide intuitive
interfaces and fast, practical, and mathematically-correct methods for
topological data analysis. 

Basic Use Cases
---------------

The repository contains many examples at 
http://nbviewer.jupyter.org/github/geomdata/gda-public/tree/master/examples

To dive into the tools, here are a few good examples to get started:

    * Multi-dimensional Data
       * Using the `multidim.PointCloud` class to compute persistence diagrams of multidimensional data. See example-pointcloud_
       * Using the `multidim.covertree.CoverTree` class to compute Cover Trees with the Friends algorithm. See example-covertree_
       * Using the `multidim.models.CDER` class for supervised machine learning using the CDER algorithm. See example-cder_

    * Time-Series Data
       * Using the `timeseries.Signal` class to compute persistence of time-series data. See example-signal-pers0_
       * Using the `timeseries.SpaceCurve` class to build and study the geometry of trajectories. See example-trajectories_
       * Using the `timeseries.SpaceCurve` class to compute signature curves.  See example-sigcurves_
       * Using mollifiers to clean up time-series data with gaps or jumps. See example-mollification_
 
Startup
-------

.. toctree::

    README
    examples_README
    README_ADVANCED
    CONTRIBUTING
    LICENSE
    
General Framework
-----------------
 .. toctree::
    :maxdepth: 1
    
    multidimensional_data
    timeseries_data
    topological_constructions


Fast Algorithms
===============
Performance-sensitive algorithms are written in [Cython].
Usually, only core developers need to access these methods directly, as most
users will use the pure Python interfaces for convenience and clarity.

.. autosummary::
    :toctree:

    multidim.fast_algorithms
    timeseries.fast_algorithms
    timeseries.curve_geometry
    homology.dim0
    homology.dim1

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`

.. toctree::
    :maxdepth: 1
    
    all_methods

References 
==========

.. _example-pointcloud: http://nbviewer.jupyter.org/github/geomdata/gda-public/blob/master/examples/example-pointcloud.ipynb
.. _example-covertree: http://nbviewer.jupyter.org/github/geomdata/gda-public/blob/master/examples/example-covertree.ipynb
.. _example-cder: http://nbviewer.jupyter.org/github/geomdata/gda-public/blob/master/examples/example-cder.ipynb
.. _example-signal-pers0: http://nbviewer.jupyter.org/github/geomdata/gda-public/blob/master/examples/example-signal-pers0.ipynb
.. _example-trajectories: http://nbviewer.jupyter.org/github/geomdata/gda-public/blob/master/examples/example-trajectories.ipynb
.. _example-sigcurves: http://nbviewer.jupyter.org/github/geomdata/gda-public/blob/master/examples/example-sigcurves.ipynb
.. _example-mollification: http://nbviewer.jupyter.org/github/geomdata/gda-public/blob/master/examples/example-mollification.ipynb
.. _Cython: http://cython.org
