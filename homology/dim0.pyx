# cython: language_level=3, boundscheck=True, linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

r""" 
This Cython module contains the core algorithms for 0-dimension topological
algorithms. In particular, MergeTree for 0-dimensional persistence.

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

 - :func:`homology.dim0.root`
 - :func:`homology.dim0.connected`
 - :func:`homology.dim0.merge`    

Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE
"""

"""
The following stanza is for docstring discovery in Cython.
>>> import numpy as np
>>> from homology.dim0 import *
>>> print("test")
test

"""

# boilerplate for efficient NumPy indexing
import numpy as np
cimport numpy as np
ctypedef np.int64_t NINT_t
ctypedef np.float64_t NDBL_t
ctypedef np.uint8_t NBIT_t

import collections

cpdef void all_roots(np.ndarray[NINT_t] components):
    r""" Reference each vertex directly to its representative root, from
    MergeTree.
    
    This is useful at the end of a MergeTree process, to make sure that
    components can be looked up quickly.

    Parameters
    ----------
    components : :class:`numpy.ndarray`
        The current component look-up table, a NumPy array of
        :class:`numpy.int64` indices.
    """
    cdef NINT_t i
    cdef NINT_t N = components.shape[0]

    for i in range(N):
        components[i] = root(components, i)
    pass

cdef NINT_t root(np.ndarray[NINT_t] components, NINT_t i):
    r""" Find the root of the component containing given vertex.
    
    Parameters
    ----------
    components : :class:`numpy.ndarray`
        The current component look-up table, a NumPy array of
        :class:`numpy.int64` indices.
    i : :class:`numpy.int64`
        The index of the vertex to examine.
    
    Returns
    -------
    j : 'class`:numpy.int64`
        The index of the root vertex.
    
    """
    # use a loop instead of recursion to avoid python's depth limit
    while i != components[i]:
        i = components[i]
    return i

cpdef int connected(np.ndarray[NINT_t] components, NINT_t i, NINT_t j):
    r""" Determine whether two vertices are in the same connected component.
    This simply calls :func:`root` on each vertex. 
    
    Parameters
    ----------
    components : :class:`numpy.ndarray`
        The current component look-up table, a NumPy array of
        :class:`numpy.int64` indices.
    i : :class:`numpy.int64`
        The index of a vertex.
    j : :class:`numpy.int64`
        The index of a vertex.
    
    Returns
    -------
    `bool`
    """

    return root(components, i) == root(components, j)

cdef int merge(np.ndarray[NINT_t] components, NINT_t i, NINT_t j):
    r""" merge (that is, join/union) two components """
    cdef NINT_t p, q
    p = root(components, i)
    q = root( components, j)
    components[q] = p
    pass

cpdef unionfind(object myobject, NDBL_t cutoff, NINT_t diagonal, NINT_t index0, NINT_t index1):
    r""" Apply the UnionFind algorithm to compute zero-dimensional persistence
    diagram.
    Connected components are born at the height of a vertex.
    Connected components merge/die at the height of the joining edge.

    Parameters
    ----------
    myobject : :class:`multidim.SimplicialComplex`, :class:`multidim.PointCloud`,
        or :class:`timeseries.Signal`
        The algorithm requires *heights* on the vertices and heights on the edges.
    cutoff : :class:`numpy.float64`
        Stop computing persistence at height :code:`cutoff`.
    diagonal: whether to return the entries with persistence 0 (diagonal)
    index0: stop when index0 and index1 are connected.  If equal, skip.
    index1: stop when index0 and index1 are connected.  If equal, skip.

    Returns
    -------
    birth_index : :class:`numpy.ndarray`
    death_index : :class:`numpy.ndarray`
    birth_height : :class:`numpy.ndarray`
    death_height : :class:`numpy.ndarray`
    mergetree : `stdtypes.dict`
        The mergetree is a dictionary keyed by the index of vertices where a
        merge occurred.  The values of the dictionary are the
        previous representatives that were merged. 
    

    Notes
    ------
    If the input object has :math:`n` edges, then this functiomn calls 
    :func:`homology.dim0.merge` a total of :math:`n` times, and it calls 
    :func:`persistence.fast_algorithms.root` a total of :math:`2n` times.
   
    References
    ----------
    .. [1]
        H. Edelsbrunner and J. Harer, Computational Topology: An
        Introduction. American Mathematical Soc., 2010.
    """
    cdef NINT_t idx, n, i, a, b, e_max, previous_merge_a, previous_merge_b
    cdef NDBL_t a_hgt, b_hgt, e_hgt
     
    cdef np.ndarray[NINT_t] components
    cdef np.ndarray[NDBL_t] verts_hgt
    cdef np.ndarray[NINT_t] edges_src
    cdef np.ndarray[NINT_t] edges_dst
    cdef np.ndarray[NDBL_t] edges_hgt
    cdef np.ndarray[NINT_t] edges_max
    cdef np.ndarray[NBIT_t, cast=True] verts_pos
    cdef np.ndarray[NBIT_t, cast=True] edges_pos

    if myobject.__class__.__module__ == "timeseries":
        components = myobject.components.values
        verts_hgt = myobject.vertices['height'].values
        verts_pos = np.ones_like(verts_hgt, dtype='bool')
        edges_src = myobject.edges['src'].values
        edges_dst = myobject.edges['dst'].values
        edges_hgt = myobject.edges['height' ].values
        edges_max = myobject.edges['max'].values
        n = edges_src.shape[0]
        edges_pos = np.zeros(shape=(n,), dtype='bool') # not needed.

    elif myobject.__class__.__module__ == "multidim":
        components = myobject.stratum[0]['rep'].values
        verts_hgt = myobject.stratum[0]['height'].values
        verts_pos = myobject.stratum[0]['pos'].values
        edges_src = myobject.stratum[1]['bdy0'].values
        edges_dst = myobject.stratum[1]['bdy1'].values
        edges_hgt = myobject.stratum[1]['height'].values
        edges_pos = myobject.stratum[1]['pos'].values
        n = edges_src.shape[0]
        edges_max = np.zeros(shape=(n,), dtype=np.int64)
    
        for idx in range(n):
            e_src = edges_src[idx]
            e_dst = edges_dst[idx]
            # Who's on top?  Identify the death location.
            if verts_hgt[e_src] < verts_hgt[e_dst]:
                e_max = e_dst
            elif verts_hgt[e_src] > verts_hgt[e_dst]:
                e_max = e_src
            elif e_src < e_dst:
                e_max = e_dst
            elif e_src > e_dst:
                e_max = e_src
            else:
                raise ValueError("Two vertices have the same height and index!")
            edges_max[idx] = e_max
        
    else:
        raise TypeError("Input to unionfind must be a multidim.SimplicialComplex or timeseries.Signal.")

    cdef np.ndarray[NINT_t] merges = np.arange(components.shape[0], dtype=np.int64)
    cdef NINT_t absbirth_i = verts_hgt.argmin()
    cdef NINT_t absdeath_i = verts_hgt.argmax()
    n = edges_src.shape[0]
 

    assert not (np.isnan(absbirth_i) or np.isnan(absdeath_i)),\
        "Yikes. We lost our bounds. -- {} {}".format(absbirth_i,absdeath_i)
    
    #if absbirth_i == absdeath_i: 
    #    return np.array([]), np.array([]), np.array([]), np.array([]), dict([])
    birth_i = []
    death_i = []
    death = []
    birth = []

    assert edges_src.size == edges_dst.size == edges_max.size == edges_hgt.size,\
        "Incompatible sizes for edge data."

    mergetree = []

    cdef NINT_t ncomps = components.shape[0]
    

    for idx in range(n):
        if ncomps <= 1: break
        if index0 != index1:
            if connected(components, index0, index1):
                break

        e_hgt = edges_hgt[idx]
        if 0 <= cutoff <= e_hgt: break

        e_src = edges_src[idx]
        e_dst = edges_dst[idx]
        e_max = edges_max[idx]
        
        a = root(components, e_src)
        b = root(components, e_dst)
        a_hgt = verts_hgt[a]
        b_hgt = verts_hgt[b]
         
        if a_hgt < b_hgt:
            merge(components, a, b)  # point b to a (so a is the representative)
            verts_pos[b] = False
            edges_pos[idx] = False ## for RCA1
            # Unless we want to see the diagonal, don't save the instant-death case.
            if diagonal == 1 or b_hgt != e_hgt:
                birth_i.append(b)
                death_i.append(e_max)
                birth.append(b_hgt)
                death.append(e_hgt)
                # track/make mergetree.  Every min and max so far should be
                # marked.  
                # The roots are a,b.
                # They were most recently involved at prev_merge_a,b
                # those roots and those deaths to e_max. 
                # additionally, mark the top of the current edge, so that
                # e_max always points to itself. 
                previous_merge_a = root(merges, a)
                previous_merge_b = root(merges, b)
                merges[previous_merge_a] = e_max
                merges[previous_merge_b] = e_max
                merges[a] = e_max
                merges[b] = e_max
                merges[e_max] = e_max
                mergetree.append( (e_max, (previous_merge_a, previous_merge_b)) )
            ncomps -= 1

        elif b_hgt < a_hgt:
            merge(components, b, a)
            verts_pos[a] = False
            edges_pos[idx] = False ## for RCA1
            if a_hgt != e_hgt:
                birth_i.append(a)
                death_i.append(e_max)
                birth.append(a_hgt)
                death.append(e_hgt)
                previous_merge_a = root(merges, a)
                previous_merge_b = root(merges, b)
                merges[previous_merge_a] = e_max
                merges[previous_merge_b] = e_max
                merges[a] = e_max
                merges[b] = e_max
                merges[e_max] = e_max
                mergetree.append( (e_max, (previous_merge_a, previous_merge_b)) )
            ncomps -= 1
        
        elif a < b:
            merge(components, a, b)
            verts_pos[b] = False
            edges_pos[idx] = False ## for RCA1
            if b_hgt != e_hgt:
                birth_i.append(b)
                death_i.append(e_max)
                birth.append(b_hgt)
                death.append(e_hgt)
                previous_merge_a = root(merges, a)
                previous_merge_b = root(merges, b)
                merges[previous_merge_a] = e_max
                merges[previous_merge_b] = e_max
                merges[a] = e_max
                merges[b] = e_max
                merges[e_max] = e_max
                mergetree.append( (e_max, (previous_merge_a, previous_merge_b)) )
            ncomps -= 1

        elif b < a:
            merge(components, b, a)
            verts_pos[a] = False
            edges_pos[idx] = False ## for RCA1
            if a_hgt != e_hgt:
                birth_i.append(a)
                death_i.append(e_max)
                birth.append(a_hgt)
                death.append(e_hgt)
                previous_merge_a = root(merges, a)
                previous_merge_b = root(merges, b)
                merges[previous_merge_a] = e_max
                merges[previous_merge_b] = e_max
                merges[a] = e_max
                merges[b] = e_max
                merges[e_max] = e_max
                mergetree.append( (e_max, (previous_merge_a, previous_merge_b)) )
            ncomps -= 1
        
    birth_i.append(absbirth_i)
    death_i.append(absdeath_i)
    birth.append(verts_hgt[absbirth_i])
    death.append(verts_hgt[absdeath_i])
    assert ncomps > 0, "{} comps".format(ncomps)
    return np.array(birth_i, dtype=np.int64), np.array(death_i, dtype=np.int64), np.array(birth, dtype=np.float64), np.array(death, dtype=np.float64), dict(mergetree)


cpdef mkforestDBL(np.ndarray[NINT_t, ndim=1] idents,
                 np.ndarray[NDBL_t, ndim=1] begins,
                 np.ndarray[NDBL_t, ndim=1] closes):
    r""" Make inclusion tree and parentage linked lists by walking across
        This method is for :class:`np.float64` arrays (for indexing speed).
        open/closed parentheses.  
        
        Parameters
        ----------
        idents : :class:`np.ndarray` 
            The index of a Pandas DataFrame, for identifying locations
        begins : :class:`np.ndarray` 
            The lower heights of a persistence bar.
        closes : :class:`np.ndarray` 
            The upper heights of a persistence bar.

        Returns
        -------
        forest : :class:`collections.defaultdict`
        forest_parents : `dict`
    """

    cdef np.ndarray[NINT_t, ndim=1] index = begins.argsort()

    past_index = [] #collections.deque([])
    past_close = [] #collections.deque([])

    tree=collections.defaultdict(set)
    parents=dict()
    cdef NINT_t i, my_ident
    cdef NDBL_t my_begin, my_close

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
