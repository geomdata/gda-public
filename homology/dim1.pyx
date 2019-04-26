# coding=utf-8
# cython: language_level=3, boundscheck=True, linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

r""" 
This Cython module contains the core algorithms for 1-dimensional topological
algorithms. In particular, RCA1 and Ripser for 1-dimensional persistence.

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

 - (currently none)

Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE
"""

"""
The following stanza is for docstring discovery in Cython.
>>> import numpy, pandas
>>> from homology.dim1 import *
>>> print("test")
test

"""

## boilerplate for efficient np indexing
import numpy as np 
cimport numpy as np
cimport cython
ctypedef np.int64_t NINT_t
ctypedef np.float64_t NDBL_t
ctypedef np.uint8_t NBIT_t

# for efficient indexing
import itertools
import collections

cpdef dict intersect_links(dict link1, dict link2):
    int_keys = set(link1.keys()) & set(link2.keys()) # set() for Python 2
    return dict([(k, (link1[k], link2[k])) for k in int_keys ])

cpdef reduce_matrix(columnList):
    """Reduce the matrix of distances
    """
    # Set max_row equal to the highest index in any column.
    last_indices = [ c[-1] for c in columnList if c ]
    cdef NINT_t max_row = -1
    if last_indices:
        max_row = max(last_indices)
    
    low = dict( [ (i,None) for i in range(max_row+1) ] )
    # low[i] = j means the jth column has its lowest 1 in row i
    # and the matrix has been reduced to the left of this column
    # and no other column with index less that j has a lowest 1 in row i
    # low[i] = None if no column has its lowest 1 in the ith row.
    for counter, column in enumerate(columnList):
        if column:
            while column and (low[column[-1]] is not None):
                columnList[counter] = sorted(list(set(column) ^ set(columnList[low[column[-1]]])))
                column = columnList[counter]
        if column:
            low[column[-1]] = counter
    return columnList

cpdef rca1(edges, NDBL_t cutoff=-1.0, NINT_t begin=0):
    """ Perform Rips Collapse Algorithm for one-dimensional
     persistence to an edge list.

    Parameters
    ----------
    edges : :class:`pandas.DataFrame`
        Edge list as Dataframe. edges.columns.values =
        array(['bdy0', 'bdy1', 'pos', 'rep', 'height'], dtype=object)
    cutoff : double

    begin : int

    Notes
    -----
    Algorithm developed by Paul Bendich, Bryan Jacobson, John Harer, and
    Jurgen Slaczedek and documented in internal communication. [1]_

    Returns
    -------
    reduce_matrix(columnList) : list
        Reduced distance matrix as a column list of binary values
    columnEdgeIndex : list

    edge_idx : list


    References
    ----------
    .. [1]
        Paul Bendich, Bryan Jacobson, John Harer, and Jurgen Slaczedek,
        RipsCollapse and Local Spherical Distance. Preprint, 2014.

    See Also
    --------
    :func:`reduce_matrix`, :class:`SimplicialComplex`


    Examples
    --------
    We consider the following example from [1]_.

    >>> edges = pandas.DataFrame({'height':[ 1.0, 1.0, 1.0],
    ...                           'pos': [True, True, True],
    ...                           'rep' : [0, 1, 2],
    ...                           'bdy0': [0, 1, 2],
    ...                           'bdy1': [1, 2, 0]})

    """
    cdef np.ndarray[NINT_t] edges_idx = edges.index.values
    cdef np.ndarray[NDBL_t] edges_hgt = edges['height'].values
    cdef np.ndarray[NBIT_t, cast=True] edges_pos = edges['pos'].values
    cdef np.ndarray[NINT_t, ndim=2] edges_bdy  = edges[['bdy0','bdy1']].values

    DAGs = collections.defaultdict(set)
    links = collections.defaultdict(dict)  
    columnList = [] #collections.defaultdict(set)
    rowVector = set() # sorted vector of edge indices -- rows of matrix to reduce
    columnEdgeIndex = [] # stores the index of the edge which creates the column
    cdef NINT_t a, b, i, j, ii, jj, e0, count, e1, edge_idx
    cdef np.ndarray[NINT_t] allkeys, U, keys

    edge_idx = -1 ## just to avoid compiler warning
    for edge_idx in range(begin, edges_hgt.shape[0]):
        assert edge_idx == edges_idx[edge_idx]
        if 0 <= cutoff and cutoff <= edges_hgt[edge_idx]: break
        a = edges_bdy[edge_idx][0]
        b = edges_bdy[edge_idx][1]
        
        if edges_pos[edge_idx]:
            lowerLinkTriples = intersect_links( links[a], links[b] )
            if not lowerLinkTriples:
                rowVector.add(edge_idx)
                initDAG = {edge_idx} 
                DAGs[edge_idx] = initDAG
            else:
                allkeys = np.array(list(lowerLinkTriples.keys()))
                count = len(lowerLinkTriples)
                U = np.arange(count, dtype=np.int64)
                #U = dict( [ (i,i) for i in lowerLinkTriples.keys() ] ) 
                for i,j in itertools.combinations(U, 2):
                #for i,j in itertools.combinations(lowerLinkTriples.keys(),2):
                    if count <= 1: break
                    if allkeys[j] in links[allkeys[i]].keys():
                        ii = i
                        jj = j;
                        while U[ii]!=ii: ii = U[ii]
                        while U[jj]!=jj: jj = U[jj]
                        if jj > ii:
                            U[jj] = ii
                            count-= 1
                        elif ii > jj:
                            U[ii] = jj
                            count -= 1 
                # Find list of representatives for each component of lower link
                keys = allkeys[U == np.arange(U.shape[0])]
                #keys = sorted([ k for k in lowerLinkTriples.keys() if U[k] == k ])
                # Now we use this information to build dagList and columns of M12
                assert keys.shape[0] > 0, "Bad Reps."

                # First representative: use to build dagList for *itE
                k = keys[0]
                e0 = lowerLinkTriples[k][0]  # Edge from first vertex of edge to first new vertex
                e1 = lowerLinkTriples[k][1]  # Edge from second vertex of edge to first new vertex
                if edges_pos[e0] and edges_pos[e1]:
                    DAGs[edge_idx] = DAGs[e0] ^ DAGs[e1]
                elif edges_pos[e0] and (not edges_pos[e1]):
                    DAGs[edge_idx] = DAGs[e0]
                elif (not edges_pos[e0]) and edges_pos[e1]:
                    DAGs[edge_idx] = DAGs[e1]
                else:
                    edges_pos[edge_idx]=False
                
                for k in keys[1:]:
                    tempList = DAGs[edge_idx]
                    e0 = lowerLinkTriples[k][0]  # Edge from first vertex of edge to new vertex
                    e1 = lowerLinkTriples[k][1]  # Edge from second vertex of edge to new vertex
                    if edges_pos[e0]:
                        tempList = tempList ^ DAGs[e0]
                    if edges_pos[e1]:
                        tempList = tempList ^ DAGs[e1]
                   
                    columnList.append(sorted(list(tempList)))
                    columnEdgeIndex.append(edge_idx)
                    
        
        # Processing done now augment links
        links[a][b] = edge_idx
        links[b][a] = edge_idx
    
    return reduce_matrix(columnList), columnEdgeIndex, edge_idx


