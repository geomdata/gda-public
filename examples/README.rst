Example Worksheets
==================

The ``examples/`` contains Jupyter notebooks that demonstrate various features.

**Note** that jupyter should be launched in a conda environment, and it should be
launched from the parent folder of this one to enusre that paths are correct.
.. code::

    (gda_env) bash$ jupyter notebook --notebook-dir /path/to/gda-public/examples/


Also, if you have *not* installed gda-public using pip, then Python will be executed from ``/path/to/gda-public/examples`` while
the modules are built in ``/path/to/gda-public``, and you will need to update the
path in your Jupyter notebook in order to import the modules.

.. code::
    
    In [1]: import sys
    In [2]: sys.path.append("..")
    In [3]: print(sys.path)
    [ <some things>, '..']



