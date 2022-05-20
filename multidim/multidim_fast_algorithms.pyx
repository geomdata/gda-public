# coding=utf-8
# cython: language_level=3, boundscheck=True, linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

r""" 
This Cython module contains core algorithms for multidimensional data.
In particular, some distance caching and the cover-tree friends matching.

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

 - :func:`is_partition_bool`
 - :func:`is_partition_list`

Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE
"""

""" 
This stanza is for pytest discovery
>>> import numpy, pandas
>>> from multidim.fast_algorithms import *

"""

### Allow us to use np indexing and datatypes
import numpy as np 
cimport numpy as np
ctypedef np.int64_t NINT_t
ctypedef np.float64_t NDBL_t
ctypedef np.uint8_t NBIT_t

# For convenient indexing
import itertools

from scipy.spatial.distance import euclidean, cdist

cpdef NINT_t check_heights(object myobject, NINT_t dim=1):
    """ Verify that cells of dimension `dim` have weight greater than
    their boundaries of dimension `dim - 1`.

    Returns "-1" if ALL GOOD.
    Otherwise, returns the index of the first bad cell.
    """

    cdef NINT_t idx
    cdef np.ndarray[NINT_t, ndim=2] cell_bdy
    cdef np.ndarray[NDBL_t] bdy_hgt, cell_hgt

    if dim <= 0:
        raise ValueError("Wrong dimension?  0 cells have no boundary.")

    if myobject.__class__.__module__ == "multidim":

        bdy_hgt = myobject.stratum[0]['height'].values
        cell_bdy = myobject.stratum[1].filter(regex='^bdy[0-9]+').values
        cell_hgt = myobject.stratum[1]['height'].values
        for idx in range(cell_hgt.shape[0]):
            if bdy_hgt[cell_bdy[idx,:]].max() >= cell_hgt[idx]:
                return idx

        return np.int64(-1)

    else:
        raise TypeError("Input must be a multidim.SimplicialComplex.")


cpdef label_means(
        np.ndarray[NDBL_t, ndim=2] coords,
        np.ndarray[NINT_t, ndim=1] labels,
        np.ndarray[NDBL_t, ndim=1] weights,
        np.ndarray[NINT_t, ndim=1] label_set):
    """ Given a list of points, a list of labels, and a list of weights,
    compute the weighted mean for each label. """
    cdef NINT_t l, i, N, d, L, label
    N = coords.shape[0]
    d = coords.shape[1]
    L = label_set.shape[0]
    cdef np.ndarray[NINT_t, ndim=1] label_matches
    cdef np.ndarray[NDBL_t, ndim=1] label_weights
    cdef np.ndarray[NDBL_t, ndim=2] label_coords
    cdef np.ndarray[NDBL_t, ndim=1] weight_by_label = np.ndarray(shape=(L,), dtype=np.float64)
    cdef np.ndarray[NDBL_t, ndim=2] mean_by_label = np.ndarray(shape=(L,d), dtype=np.float64)
    for l in range(L):
        label = label_set[l]
        label_matches = np.where(labels == label)[0]
        label_weights = weights[label_matches]
        weight_by_label[l] = label_weights.sum()
        label_coords =  coords[label_matches, :].copy()
        for i in range(d):
            label_coords[:,i] = label_coords[:,i]*label_weights/weight_by_label[l]
        
        mean_by_label[l,:] = np.sum(label_coords, axis=0)
    return mean_by_label, weight_by_label

cpdef np.ndarray[NDBL_t, ndim=1] label_weights(
        np.ndarray[NBIT_t, ndim=1, cast=True] pointset,
        np.ndarray[NINT_t, ndim=1] pointwise_labels,
        np.ndarray[NDBL_t, ndim=1] pointwise_weight,
        np.ndarray[NINT_t, ndim=1] label_set):
    """ Weights-per-label of a subset with labels and weights. """
    L = label_set.shape[0]
    cdef np.ndarray[NDBL_t, ndim=1] weight_per_label = np.zeros(shape=(L,), dtype=np.float64)
    if np.count_nonzero(pointset) == np.int64(0):
        return weight_per_label

    cdef NINT_t test_label
    for test_label in label_set:
        point_is_label = pointset & (pointwise_labels == test_label)
        weight_per_label[test_label] = pointwise_weight[point_is_label].sum()

    return weight_per_label

def is_partition_bool(bigblock, list_of_small_blocks):
    """ Check that bigblock is partitioned by list_of_small_blocks """
    myunion = np.zeros_like(bigblock)
    for block in list_of_small_blocks:
        myunion = myunion | block
    if not np.all(myunion == bigblock):
        return False
 
    for block1, block2 in itertools.combinations(list_of_small_blocks, 2):
        if np.any(block1 & block2):
            return False
    return True

def is_partition_list(biglist, list_of_lists):
    """ You have a big list and a list of small lists. Make sure that 
    all the small lists together make the big list exactly. """
    biglist2 = []
    for l in list_of_lists:
        biglist2.extend(l)
    return sorted(list(biglist)) == sorted(biglist2)

cpdef NDBL_t entropy(np.ndarray[NDBL_t] bins):
    """
    Compute the base-N entropy of normalized bins
    [bin0, bin1, ..., bin(N-1)]
    
    You should prenormalize frequencies so that bins.sum() == 1.0
    """
    cdef NINT_t N = bins.shape[0]
    cdef np.ndarray[NDBL_t] goodbins = bins[bins > 0.0]
    cdef np.ndarray[NDBL_t] p_logp = goodbins*np.log(goodbins)/np.log(N)
    #p_logp[np.isnan(p_logp)] = NDBL(0.) # 0log0 = 0*inf = nan --> 0
    return - p_logp.sum() 

cpdef gaussian_fit_wt(np.ndarray[NDBL_t, ndim=2] cloud, np.ndarray[NDBL_t] weights, center=None):
    """ Fit a normalized Gaussian to this cloud using singular value
    decomposition, with weights on the points.

    See Also:
    ---------
    :func:`gaussian_fit`

    """
    cdef np.ndarray[NDBL_t, ndim=1] mean, s
    cdef NINT_t N, totwt
    cdef np.ndarray[NDBL_t, ndim=2] u, v
    
    if center is None:
        mean = cloud.mean(axis=0)
    elif hasattr(center, '__len__') and len(center) == cloud.shape[1]:
        mean = np.array(center)
    elif isinstance(center, np.integer):
        mean = cloud[center,:]
    else:
        raise ValueError("center must be integer or tuple or None.")
    
    N = cloud.shape[0] # for bias of sample distribution.

    totwt = np.sum(weights)/(N-1)  # correct weighting???
    svd = np.linalg.svd((cloud-mean)*np.sqrt(weights*totwt/(N-1)), full_matrices=False)
    u = svd[0]
    s = svd[1]
    v = svd[2]
    
    return mean, s, v


cpdef gaussian_fit(np.ndarray[NDBL_t, ndim=2] cloud, center=None):
    """ Fit a normalized Gaussian to this cloud using singular value decomposition.

    Parameters
    ----------
    cloud : :class:`np.ndarray`
        Point cloud as np array.
    center : variable
        If center is None (default), we find the best Gaussian with free mean.
        If center is given as an integer, use the point with that index as the mean of the Gaussian.
        If center is given as a tuple or array, use that coordinate point as the mean of the Gaussian.

    Returns
    -------
    mean : :class:`np.ndarray`
    s : :class:`np.ndarray`
        singular values
    v : { (..., N, N), (..., K, N) } :class:`np.ndarray`
        right singular vectors from SVD

    See Also
    --------
    :func:`np.linalg.svd`, :func:`gaussian`

    Notes
    -----
    You can feed this to :func:`gaussian`.

    References
    ----------
    .. [1]
        R. Christensen, Plane Answers to Complex Questions: The Theory of
        Linear Models. Springer Science & Business Media, 2011.

    """
    cdef np.ndarray[NDBL_t, ndim=1] weights
    cdef NINT_t N
    cdef np.ndarray[NDBL_t, ndim=2] u, v
    
    if center is None:
        mean = cloud.mean(axis=0)
    elif hasattr(center, '__len__') and len(center) == cloud.shape[1]:
        mean = np.array(center)
    elif isinstance(center, np.integer):
        mean = cloud[center,:]
    else:
        raise ValueError("center must be integer or tuple or None.")
    
    N = cloud.shape[0] # for bias of sample distribution.

    svd = np.linalg.svd((cloud-mean)/np.sqrt(N-1.0), full_matrices=False)
    u = svd[0]
    s = svd[1]
    v = svd[2]
    
    return mean, s, v


cpdef np.ndarray[NDBL_t, ndim=1] gaussian(
      np.ndarray[NDBL_t, ndim=2] sample_points,
      np.ndarray[NDBL_t, ndim=1] mean,
      np.ndarray[NDBL_t, ndim=1] std,
      np.ndarray[NDBL_t, ndim=2] rotation):
    """ Generate value for a multivariate Gaussian PDF.
    """
    cdef np.ndarray[NDBL_t, ndim=2] x = rotation.dot((sample_points-mean).T)
    cdef NINT_t n = mean.shape[0]
    assert n == x.shape[0]
    cdef NINT_t count = x.shape[1]
    cdef NINT_t i
    cdef np.ndarray[NDBL_t, ndim=1] exponent = np.ndarray(shape=(count,), dtype=np.float64)
    for i in range(count):
        # remember that operations are broadcast, not mathematical!
        exponent[i] = -0.5*(x[:,i]**2/std**2).sum()
    cdef NDBL_t denom = np.abs(std.prod())*np.sqrt((2.0*np.pi)**n)
    return np.exp(exponent)/denom


cpdef edges_from_dists(np.ndarray[NINT_t] idx, np.ndarray[NDBL_t, ndim=2] dists, NDBL_t cutoff):
    cdef NINT_t n = idx.shape[0]
    cdef NINT_t n2 = n*(n-1)//2
    cdef np.ndarray[NINT_t, ndim=2] bdy = np.ndarray(shape=(n2,2), dtype=np.int64)
    cdef np.ndarray[NDBL_t] val = np.ndarray(shape=(n2,), dtype=np.float64)
    cdef np.ndarray[NBIT_t, cast=True] pos = np.ndarray(shape=(n2,), dtype=np.uint8)
    cdef NINT_t ct, i,j
    for ct,(i,j) in enumerate(itertools.combinations(range(n),2)):
        bdy[ct][0]=idx[i]
        bdy[ct][1]=idx[j]
        val[ct]=dists[i,j]
        pos[ct]=True
    
    cdef np.ndarray[NINT_t] sortby = val.argsort()
    val = val[sortby]
    
    cdef NINT_t end
    if cutoff >= 0.0:
        end = np.searchsorted(val,cutoff)
        sortby = sortby[:end]
        val = val[:end]
    pos = pos[sortby]
    bdy = bdy[sortby]
    return val,pos,bdy #pass #return src,dst,val,pos

cpdef NBIT_t covertree_befriend321(object coverlevel, object prev_level,
                                 NINT_t pre_i, 
                                 np.ndarray[NINT_t] f3s):
    """
    Set friends3 using inheritance filter 

    level l-1    pre_i --->-- pre_j 
                    |   f3    |
                pre ^         v  suc
                    |         | 
    level l        ca        cb
    
    """
    cdef np.ndarray[NINT_t] suc_i = prev_level.successors[pre_i]
    cdef NINT_t j, pre_j, a, b, ca, cb, p
    cdef NINT_t nf3s = f3s.shape[0]
    cdef NDBL_t T3 = coverlevel.T3
    cdef NDBL_t T2 = coverlevel.T2
    cdef NDBL_t T1 = coverlevel.T1
    cdef np.ndarray[NDBL_t, ndim=2] dists
    cdef np.ndarray[NINT_t, ndim=2] pairs3

    
    for j in range(nf3s):
        pre_j = f3s[j]
        if pre_i <= pre_j:
            suc_j = prev_level.successors[pre_j]

            #assert suc_i.size > 0 and suc_j.size > 0
            dists = distance_cache_None(suc_i, suc_j, coverlevel.covertree.coords) 
            pairs3 = np.array(np.where(dists <= T3)).T
            #assert dists.shape[0] == suc_i.shape[0]
            #assert dists.shape[1] == suc_j.shape[0]
            if pairs3.shape[0] > 0:
                for p in range(pairs3.shape[0]):
                    a = pairs3[p, 0]
                    b = pairs3[p, 1]
                    ca = suc_i[a]
                    cb = suc_j[b]
                    coverlevel.friends3[ca].append(cb)
                    coverlevel.friends3[cb].append(ca)
                    if dists[a, b] <= T2:
                        coverlevel.friends2[ca].append(cb)
                        coverlevel.friends2[cb].append(ca)
                        if dists[a, b] <= T1:
                            coverlevel.friends1[ca].append(cb)
                            coverlevel.friends1[cb].append(ca)
    return np.uint8(1)

cpdef np.ndarray[NINT_t] covertree_adopt_or_liberate(object coverlevel, 
                                         object prev_level,
                                         NINT_t orphan_index):
    # remove the orphan from the parents' list of children.
    
    r"""
    adopt an orphan using type-1 friends. 


    level l-1               old_grd --->-- old_f1 
                              |      f1     |
                          pre ^             v  suc
                              |             | 
    level l   orphan -->-- deadbeat     new_guardian
                      grd

    if no such new_parent exists, then liberate the orphan to be a new adult.
    
    """
    cdef np.ndarray[NINT_t] cg = coverlevel.guardians
    cdef NINT_t deadbeat = cg[orphan_index]
    cdef NINT_t K = coverlevel.children[deadbeat].shape[0] 
    coverlevel.children[deadbeat] = np.setdiff1d(coverlevel.children[deadbeat],
                                                np.array([orphan_index], dtype=np.int64),
                                                assume_unique=True)
    assert coverlevel.children[deadbeat].shape[0] == K - 1
    assert cg[orphan_index] == deadbeat
    cg[orphan_index] = np.int64(-1)

    cdef NINT_t old_guardian = coverlevel.predecessor[deadbeat]
    cdef NINT_t old_f1
    fosters = []
    for old_f1 in prev_level.friends1[old_guardian]:
        fosters.extend(prev_level.successors[old_f1])

    fosters = list(set(fosters))

    cdef np.ndarray[NINT_t] orphan_array = np.array([orphan_index], dtype=np.int64)
    cdef np.ndarray[NINT_t] fosters_array = np.array(fosters, dtype=np.int64)
    cdef np.ndarray[NDBL_t] new_dists 
    cdef NINT_t i, new_parent
    cdef NDBL_t d
    cdef NDBL_t R = coverlevel.radius
    cdef np.ndarray[NBIT_t, cast=True] npc 
    if fosters_array.size > 0:
        new_dists = distance_cache_None(orphan_array, fosters_array,
            coverlevel.covertree.coords).flatten()
#pointcloud.dists(suc_i, suc_j)
        #new_dists = coverlevel.pointcloud.dists(orphan_array, fosters_array)[0,:] 
        i = new_dists.argmin()
        d = new_dists[i]
        if d <= R:
            new_parent = fosters_array[i]
            cg[orphan_index] = new_parent
            coverlevel.children[new_parent] = np.union1d(coverlevel.children[new_parent], [orphan_index])
            return np.array([deadbeat, new_parent])

    # nobody claimed this orphan.  Let's promote it to an "adult" center.
    cg[orphan_index] = orphan_index
    return np.array([deadbeat, orphan_index])

cpdef np.ndarray[NINT_t, ndim=2] covertree_exchange_teens(object coverlevel, object prev_level, NINT_t ci):
    r""" For a given center of a CoverTree, determine which of its children are
    actually closer to another center. 

                  f2
    old_g ------->--------- old_f2 
     |                        |
     ^  pre              suc  v
     |                        |
  adult ci --->--- teen -<-- new_grd cj
    """
    cdef NDBL_t R = coverlevel.radius
    cdef np.ndarray[NDBL_t] childrenR  = distance_cache_None(
        np.array([ci], dtype=np.int64), coverlevel.children[ci], coverlevel.covertree.coords).flatten()
    cdef np.ndarray[NINT_t] teens = coverlevel.children[ci][childrenR > 0.5*R]
    
    cdef np.ndarray[NINT_t] old_f2s = np.array(
        prev_level.friends2[coverlevel.predecessor[ci]]) 
    cdef NINT_t i, j, cj, teen, f2
    
    #for teen in teens:
        #print(ci, teen, cci[teen], teen in np.where(cci)[0]) # == 1

    new_grd = []
    for f2 in old_f2s:
        new_grd.extend(prev_level.successors[f2])
    cdef np.ndarray[NINT_t] new_guardians = np.array(new_grd)
     
    cdef np.ndarray[NDBL_t, ndim=2] teen_dists = distance_cache_None(teens, new_guardians, coverlevel.covertree.coords) 
    cdef np.ndarray[NINT_t] teen_reassignments = teen_dists.argmin(axis=1)
   
    cdef np.ndarray[NINT_t] cgs = coverlevel.guardians
    cdef np.ndarray[NINT_t, ndim=2] results = np.ndarray(shape=(teens.shape[0], 2), dtype=np.int64)
    for i, teen in enumerate(teens):
        j = teen_reassignments[i]  # j should be np index from new_guardians
        cj = new_guardians[j]
        #print(ci, teen, cci[teen], teen in np.where(cci)[0]) # == 1
        if cj != ci:
            cgs[teen] = cj
            coverlevel.children[ci] = np.setdiff1d(coverlevel.children[ci],
                                                 [teen],
                                                 assume_unique=True)
            coverlevel.children[cj] = np.union1d(coverlevel.children[cj], [teen])
        results[i,0] = teen
        results[i,1] = cj
    return results
        
cpdef np.ndarray[NDBL_t, ndim=2] distance_cache_None(
                          np.ndarray[NINT_t] indices0, 
                          np.ndarray[NINT_t] indices1, 
                          np.ndarray[NDBL_t, ndim=2] cloud):
    """ Get bulk distances, using no storage cache. """
#    cdef np.ndarray[NDBL_t, ndim=2] cloud0 = cloud[indices0,:]
#    cdef np.ndarray[NDBL_t, ndim=2] cloud1 = cloud[indices1,:]
#    return cdist(cloud0, cloud1)

#cpdef np.ndarray[NDBL_t, ndim=2] distE(
#                         np.ndarray[NINT_t] indices0, 
#                         np.ndarray[NINT_t] indices1, 
#                         np.ndarray[NDBL_t, ndim=2] cloud):
#   """ Get bulk distances, using no storage cache. """
#    #cdef np.ndarray[NDBL_t, ndim=2] cloud0 = cloud[indices0,:]
#    #cdef np.ndarray[NDBL_t, ndim=2] cloud1 = cloud[indices1,:]
    cdef NINT_t i0, i1, k
    cdef NINT_t n0 = indices0.shape[0]
    cdef NINT_t n1 = indices1.shape[0]
    cdef np.ndarray[NDBL_t, ndim=2] output = np.ndarray(
        shape=(n0, n1), dtype=np.float64)
    for i0 in range(n0):
        for i1 in range(n1):
            output[i0, i1] = np.sqrt(np.sum((cloud[indices0[i0],:] - cloud[indices1[i1],:])**2))
    return output


cpdef np.ndarray[NDBL_t, ndim=2] distance_cache_numpy(
                           np.ndarray[NINT_t] indices0, 
                           np.ndarray[NINT_t] indices1, 
                           np.ndarray[NDBL_t, ndim=2] cloud, 
                           np.ndarray[NDBL_t, ndim=2] cache):
    """ Get bulk distances, using NumPy storage cache. """
   
    cdef NINT_t i,ind0,j,ind1
    cdef np.ndarray[NDBL_t, ndim=2] cloud0, cloud1, output

    output = np.ndarray((indices0.shape[0], indices1.shape[0]),dtype=np.float64)
    for i,ind0 in enumerate(indices0):
        for j,ind1 in enumerate(indices1):
            if cache[ind0,ind1]<0:
                cache[ind0,ind1] = euclidean(cloud[ind0,:], cloud[ind1,:])
                cache[ind1,ind0] = cache[ind0,ind1]
            output[i,j] = cache[ind0,ind1]
    return output

cpdef np.ndarray[NDBL_t, ndim=2] distance_cache_dict(
                     np.ndarray[NINT_t] indices0, 
                     np.ndarray[NINT_t] indices1, 
                     np.ndarray[NDBL_t, ndim=2] cloud, 
                     dict cache):
    """ Get bulk distances, using dictionary storage cache. """
   
    cdef NINT_t i,ind0,j,ind1,min0,max1
    cdef np.ndarray[NDBL_t, ndim=2] cloud0, cloud1, output

    output = np.ndarray((indices0.shape[0], indices1.shape[0]),dtype=np.float64)
    for i,ind0 in enumerate(indices0):
        for j,ind1 in enumerate(indices1):
            if (ind0, ind1) not in cache:
                cache[(ind0,ind1)] = euclidean(cloud[ind0,:], cloud[ind1,:])
                cache[(ind1,ind0)] = cache[(ind0, ind1)]
            output[i,j] = cache[(ind0,ind1)]
    return output

cpdef np.ndarray[NDBL_t, ndim=2] distance_cache_dok(
                     np.ndarray[NINT_t] indices0, 
                     np.ndarray[NINT_t] indices1, 
                     np.ndarray[NDBL_t, ndim=2] cloud, 
                     dict cache):
    """ Get bulk distances, using dictionary storage cache. """
   
    cdef NINT_t i,ind0,j,ind1,min0,max1
    cdef np.ndarray[NDBL_t, ndim=2] cloud0, cloud1, output

    output = np.ndarray((indices0.shape[0], indices1.shape[0]),dtype=np.float64)
    for i,ind0 in enumerate(indices0):
        for j,ind1 in enumerate(indices1):
            if (ind0, ind1) not in cache:
                cache[(ind0,ind1)] = euclidean(cloud[ind0,:], cloud[ind1,:])
                cache[(ind1,ind0)] = cache[(ind0, ind1)]
            output[i,j] = cache[(ind0,ind1)]
    return output

