README for GDA-Public
=====================

See the thorough documentation at https://geomdata.github.io/gda-public/

This contains several fundamental tools by Geometric Data Analytics Inc. [http://www.geomdata.com]
See `LICENSE` for copyright information.

The code is written in Python and Cython.  It is written for Python 3.5, but it
should work in Python 2.7 (albeit with some performance penalty).  For
stability and consistency, Anaconda is preferred.

See ``requirements.txt`` for requirements.

If you want to work in an air-gapped system, you can pre-download all of the
required packages from http://repo.continuum.io/pkgs/ This is automated in
``download.py`` and ``download-viz.py``, and is described below.

Please submit any **BUGS** to https://github.com/geomdata/gda-public/issues


Quick Start
-----------

Here are minimal instructions with no detail.  First, unpack or clone the code to ``/path/to/gda-public``.  Then

.. code::

    bash$ cd ~   # Change directory to anywhere *except* /path/to/gda-public !
    bash$ conda create --name gda_env --file /path/to/gda-public/requirements.txt python=3
    bash$ source activate gda_env
    (gda_env) bash$ pip install file:///path/to/gda-public
    (gda_env) bash$ jupyter notebook --notebook-dir /path/to/gda-public/examples

In a worksheet, try 

.. code::

    >>> import multidim, homology, timeseries
    

Building Documentation
----------------------

The package comes with thorough documentation in docstrings, accessible from
within python via the ``help( )`` command:

.. code::

    (gda_env) bash$ python
    >>> import timeseries
    >>> help(timeseries)
    >>> s = timeseries.Signal([1,2,3,4,5])
    >>> help(s)

You can build a nice HTML guide, but you need to get a copy of the SciPy Sphinx theme:

.. code::
    
    (gda_env) bash$ cd /path/to/gda-public
    (gda_env) bash$ git clone https://github.com/scipy/scipy-sphinx-theme
    (gda_env) bash$ cd doc_src
    (gda_env) bash$ ln -sf ../scipy-sphinx-theme/_theme ./
    (gda_env) bash$ cd -
    (gda_env) bash$ python setup.py build_doc_html
    
which can be viewed in a web browser at file:////path/to/gda-public/docs/build/html/index.html

You can also produce a static PDF documentation with

.. code::

    (gda_env) bash$ python setup.py build_doc_latex
    (gda_env) bash$ cd /path/to/gda-public/doc_build/latex/latex
    (gda_env) bash$ make

Or, just build the .tex file using your favorite TeX suite.


Running and Using
-----------------

This will be filled in later versions, with suggestions.

Take a look at `examples_README` and the jupyter notebooks in ``examples/``

The docstrings, accessible via ``help( )``, are always a good reference for
low-level operations.

Note!  The code contains a lot of "assert" statements, so you can speed it up
by using ``python -O``.

