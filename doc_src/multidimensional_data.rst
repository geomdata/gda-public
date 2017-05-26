Multidimensional Data
=====================


Embedded Data
-------------

Typically, data lies in some Cartesian space, :math:`\mathbb{R}^d`.

If your data can be described as :math:`N` points lying in
:math:`\mathbb{R}^d`, typically stored as an :math:`N \times d`
:class:`numpy.ndarray`, then then you probably want to use our
:class:`multidim.PointCloud` class to analyze.

.. autosummary::
    :toctree: generated/

    multidim.PointCloud
    multidim.covertree.CoverTree


Abstract Data
-------------

Sometimes, the data under consideration does not live in a specific ambient space.

.. autosummary::
    :toctree: generated/

    multidim.SimplicialComplex
    multidim.SimplexStratum
    multidim.Simplex


