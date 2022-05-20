# coding=utf-8
# cython: language_level=3, boundscheck=True, linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

r""" 
This Cython module contains core algorithms for `timeseries.Signal` data.
In particular, some feature extraction.

If a function requires its own explicit iterator in a loop, probably it should
go here.  Anything that is just a vectorized operation in NumPy or Pandas can
go in plain python elsewhere. 

See http://docs.cython.org/src/tutorial/np.html for ideas on writing efficient
code in Cython.

This module is compiled by either of these commands
 - :code:`python setup.py install`  (as called by :code:`pip` for standard installation and use)
 - :code:`python setup.py build_ext --inplace` (as run by developers for code testing)

Notes
-----
Automatically-generated online documentation will never see the C-only functions that
are defined with :code:`cdef` in Cython, as shown in 
http://cython.readthedocs.io/en/latest/src/reference/extension_types.html
Because of the limitations of Sphinx, you'll have to simply view the sourcecode
for further information. For this module, these C-only functions include 

 - (none currently)

Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE
"""

""" 
This stanza is for pytest discovery
>>> import numpy, pandas
>>> from timeseries.fast_algorithms import *

"""

## boilerplate for efficient np indexing
import numpy as np 
cimport numpy as np
cimport cython
ctypedef np.int64_t NINT_t
ctypedef np.float64_t NDBL_t
ctypedef np.uint8_t NBIT_t

import scipy.special

import collections

cpdef sample_height(signal_sigma_pair):
    """ measure H_A: U to R on Omega. """
    from concurrent.futures import ProcessPoolExecutor
    signal = signal_sigma_pair[0]
    cdef NDBL_t sigma = signal_sigma_pair[1]

    cdef NINT_t N = signal.vertices.values.shape[0]
    cdef NINT_t i,j,ij
    other = signal.sample_near(sigma=sigma)
    cdef np.ndarray[NDBL_t, ndim=2] heights = np.zeros(shape=(N, N), dtype=np.float64)
    for (i,j) in other.pers.domains.keys():
        heights[i, j] = other.interval_height((i, j))
    return heights

cpdef mkforestINT(np.ndarray[NINT_t, ndim=1] idents,
                 np.ndarray[NINT_t, ndim=1] begins,
                 np.ndarray[NINT_t, ndim=1] closes):
    """ make inclusion tree and parentage linked lists by walking across
        open/closed parentheses.  ident contains the Pandas index.
        This method is for INT arrays (for indexing speed).
        Return forest (defaultdict), parents (dict) of indices.
    """

    cdef np.ndarray[NINT_t, ndim=1] index = begins.argsort()

    past_index = [] #collections.deque([])
    past_close = [] #collections.deque([])

    tree=collections.defaultdict(set)
    parents=dict()

    for i in index:
        my_ident = idents[i]
        my_begin = begins[i]
        my_close = closes[i]
        while len(past_close) > 0 and past_close[-1] <= my_close:
            past_close.pop()
            past_index.pop()

        
        if len(past_close) > 0:
            parent = idents[past_index[-1]]
        else:
            parent = None
        
        tree[parent].add(my_ident)
        assert my_ident not in parents,\
            ValueError("Parentage not a tree?")
        parents[my_ident] = parent
        
        past_close.append(my_close)
        past_index.append(i)
    
    return tree, parents


