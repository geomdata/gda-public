r"""
Test whether all Cython files were compiled correctly.

Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE
"""

"""
doctest discovery

Examples
--------

>>> test_cython()
"""


import os
msg = "Compiled Cython out-of-date.  Please re-run 'python setup.py build_ext --inplace'"
def test_cython():
    
    try: 
        import multidim.fast_algorithms as tda
        import timeseries.fast_algorithms as barcode
        import timeseries.curve_geometry as curve_geometry
        import homology.dim0 as dim0
        import homology.dim1 as dim1

        if os.path.getmtime(tda.__file__) <= os.path.getmtime("multidim/fast_algorithms.pyx"):
            assert False, ImportError(msg)
        if os.path.getmtime(barcode.__file__) <= os.path.getmtime("timeseries/fast_algorithms.pyx"):
            assert False, ImportError(msg)
        if os.path.getmtime(curve_geometry.__file__) <= os.path.getmtime("timeseries/curve_geometry.pyx"):
            assert False, ImportError(msg)
        if os.path.getmtime(dim0.__file__) <= os.path.getmtime("homology/dim0.pyx"):
            assert False, ImportError(msg)
        if os.path.getmtime(dim1.__file__) <= os.path.getmtime("homology/dim1.pyx"):
            assert False, ImportError(msg)
    except ImportError:
        assert False, ImportError(msg)


