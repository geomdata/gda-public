Development Guidelines
======================

Repository usage
----------------

Public open-source development happens at https://github.com/geomdata/gda-public

To get started:

.. code::

    git clone -b dev git@github.com:geomdata/gda-public.git

Ongoing cleanup work should happen in the ``dev`` branch.  Specific feature
additions can happen in their own short-lived branch.  Whenever possible, the
``master`` branch should **not** be committed to directly; instead, commit
changes to ``dev``, then apply a merge request from ``dev`` to ``master``.  This way,
``dev`` is always ahead of ``master``.

The contents of ``master`` should always pass all the unit tests.  Merges from
``dev`` to ``master`` should not be accepted otherwise.

Versions of ``master`` will be tagged ``VERSION_X.Y.Z`` according to milestones.
See https://gitgub.com/geomdata/gda-public/tags

Profiling
---------

When writing new code, try to think about what a good object model would look
like, from an end-user's perspective.  Be Pythonic and avoid surprises.

Try writing the code first in pure Python to establish an object model and
input/putput format.  After it works, then use a profiling system (such as the
Python builtin ``cProfile`` or something interactive like
https://github.com/what-studio/profiling) to determine where it is slow.
Another simple way to profile is to use ``%prun`` or ``%timeit`` in IPython or
Jupyter.  Transfer the slow parts of your code to a compiled Cython .pyx file.


Style
-----

Try to follow PEP8 (https://www.python.org/dev/peps/pep-0008/). Specifically,
keep to 80-character lines, use 4 spaces (not TABs), and avoid superfluous
spacing.  Longer lines are acceptable in Cython (.pyx) code when they are due
to type declarations.

A lot of our data is arrays.  (Coordinates, point-clouds, matrices, etc.)
We use Pandas DataFrames (or Series) as a presentation layer, because it has
nice labels and indexing.  We use the underlying NumPy array as the data layer,
as it is fast, especially via Cython.  That is, the user should use ``DF =
DataFrame( )``.  Performance-sensitive code should access ``DF.index.values``
for the index set and ``DF.values`` for the actual content. 


Documentation
-------------

Comments describing functions, classes, and modules should be in docstrings.
Use #-comments only for short clarifying notes to future developers; however,
if you need #-comments in your code to understand it, it is probably too messy.

Make sure the documentation is sane, via Sphinx

.. code::

    (gda_env) bash$ cd /path/to/gda-public/
    (gda_env) bash$ python setup.py build_doc_html
    (gda_env) bash$ python setup.py build_doc_latex
    
and open a browser to http:///path/to/gda-public/docs/index.html

Testing
-------

Write tests before, during, and after writing
.. code::

    (gda_env) bash$ cd /path/to/gda-public/
    (gda_env) bash$ py.test

Simple function tests can be in docstrings, run using ``doctest`` or (preferably)
``py.test``.  More complicated tests should be written under the ``tests/``
directory.

Try to keep dependencies to the Anaconda built-in libraries.

If newer versions of Python and NumPy break this code, debug by creating a
conda env (or python venv) with different versions to compare.


Docker Build Environment
------------------------

To build a new Docker environment, get Docker running on your machine and do
this 
..code::

    bash$ cd /path/to/gda-public
    bash$ docker build -t <name-for-container> -f .dockerfile . 

The official container is at https://hub.docker.com/r/geomdata/builder/
This container is used for automated unit tests, as in ``.gitlab-ci.yml``


