r"""This module contains the essential classes for the "Cover-tree with
friends" algorithm, namely:

    - :class:`CoverTree`
    - :class:`CoverLevel`

This module also defines the constants

    - :code:`ratio_Ag` :math:`=\sqrt{2} - 1=0.414\ldots`, the inverse of the silver ratio
    - :code:`ratio_Au` :math:`=\frac{\sqrt{5} - 1}{2}=0.618\ldots`, the inverse of the golden ratio

Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE
"""

from __future__ import print_function

from copy import deepcopy
import numpy as np
import pandas as pd
from . import PointCloud
from . import fast_algorithms

from scipy.spatial.distance import cdist, pdist, squareform

from collections import OrderedDict

import collections
import logging

ratio_Ag = np.float64(0.41421356237309504880168872420969807857)  
ratio_Au = np.float64(0.61803398874989484820458683436563811772)
assert ratio_Ag**2 + 2*ratio_Ag == np.float64(1.0),\
    """pre-defined ratio_Ag does not match artithmetic.
    Try using some form of sqrt(2) - 1, which is the positive root of x**2 + 2*x == 1."""
assert ratio_Au**2 + 1*ratio_Au == np.float64(1.0),\
    """pre-defined ratio_Au does not match artithmetic.
    Try using some form of (sqrt(5) - 1)/2, which is the positive root of x**2 + x == 1."""


class CoverTree(object):
    r"""An efficient and convenient implementation of
    the "Cover Tree with Friends" algorithm.  
    This implementation follows the notation and terminology of the paper
    [CDER1]_ as carefully as possible; they were written in concert.

    A CoverTree is an Python iterator object [iter1]_ [iter2]_.
    The :func:`__next__` and :func:`__getitem__` methods yield the
    :class:`CoverLevel` with that index.  The entire "friends" algorithm happens
    in :func:`multidim.covertree.CoverTree.__next__`

    Parameters
    ----------
    pointcloud : :class:`multidim.PointCloud`
        The original data from which to construct a cover tree.  Note that the
        labeling/weighting/indexing system requires the use of
        :class:`multidim.PointCloud` input, not merely a
        :class:`numpy.ndarray`.  However, `CoverTree` ignores all of the higher 
        strata (edges, faces, and so on) of the :class:`multidim.PointCloud`.
        Only the points in stratum[0] are used.
    ratio : float
        Ratio :math:`\theta` to shrink radii by at each step. Must satisfy
        :math:`0<\theta<1`. Good values are :code:`0.5` or
        :code:`ratio_Ag` or :code:`ratio_Au`.  Default: :code:`ratio_Ag`
    exchange_teens : bool
        Should teens be exchanged at each step, using Type-2 friends?  
        Default: :code:`True`
    sort_orphans_by_mean : bool
        Should orphans be re-ordered by their proximity to weighted mean of the
        labels?  This is particularly useful for improving the cross-validation
        score of the :class:`multidim.models.CDER` classifier.  Disable for
        speed ordering of adults is irrelevant for your needs.
        Default: :code:`True`

    Yields
    ------
    :class:`multidim.covertree.CoverLevel`
        From level 0 (one ball) until all points are separated.  Each `CoverLevel` 
        is cached once computed.

    Attributes
    ----------
    pointcloud : :class:`multidim.PointCloud`
        The original dataset.
    ratio : :class:`numpy.float64`
        Ratio :math:`\theta` by which to shrink the ball radius between levels.
    _r0 : :class:`numpy.float64`
        The initial radius at level 0.
    _adult0 : :class:`numpy.int64`
        The index of the original adult.  Typically, this is the index of the
        point nearest the weighted mean of the :class:`PointCloud`
    _levels : :class:`collections.OrderedDict`
        An ordered dictionary to cache the levels computed so far, keyed by the
        index.  Typically, a user would never access this directly.  Insead, use
        :code:`covertree[i]` 
    cohort : :class:`numpy.ndarray`
        An array of :class:`numpy.int64`, which keeps track of the cohort 
        (that is, the level in the filtration) of each point.  If a point has
        not been born as an adult yet, the value is -1
    level_pointer : int
        Index of the currently referenced `CoverLevel`, for iteration purposes.
        Setting this is like using :func:`file.seek` on file objects.  Usually,
        you don't want to mess with it, but it is used internally in
        :class:`mutlidim.models.CDER` for comparing entropy between levels.
    N : int
        The number of points in :code:`self.pointcloud`
    allpoints : :class:`numpy.ndarray`
        The raw NumPY array underlying :code:`self.pointcloud`.
    

    Notes
    -----

    This section is excerpted and condensed from [CDER1]_

    **Definition**

    Let :math:`X` be a finite subset of :math:`\mathbb{R}^d`. 
    The purpose of a cover tree is to build a filtration 
    :math:`\emptyset \subset CL_0 \subset CL_1 \subset \cdots \subset CL_{\text{max}} = X`
    by covering it with balls of smaller and smaller radius centered at points in
    the set. 
    The points in :math:`CL_\ell` are called the **adults** at level :math:`\ell`.

    Specifically, a **cover tree** is a filtration of :math:`X` with the 
    following additional properties:

    - :math:`CL_0` contains a single point, :math:`a_0`.  (see :code:`_adult0`)
    - There is a radius :math:`r_0` (see :code:`_r0`) such that :math:`X` is contained in the
      ball :math:`B(a_0, r_0)` of radius :math:`r_0` around :math:`a_0`
    - There is a real number :math:`0< \theta < 1` (see :code:`ratio`) such that, for every
      :math:`\ell`, the set :math:`X` is a subset of
      :math:`\cup_{a_i \in CL_\ell} B(a_i, r_\ell)`
      where :math:`r_\ell = r_0 \theta^\ell`
    - For each :math:`\ell`, if :math:`a_i, a_j \in CL_\ell`, then 
      `\| a_i - a_j\| > r_\ell`.  No two adults lie in the same ball.
    - For each :math:`\ell`, each point :math:`x \in X` is assigned to a
      **guardian** :math:`a_i \in CL_\ell` such that :math:`x` lies in the ball
      :math:`B(a_i, r_\ell)`.  We say :math:`x` is a **child** of :math:`a_i`
      at level :math:`\ell`.  Each :math:`a_i \in CL_\ell` is its own guardian
      and its own child.
    - There is a tree structure on the (level, adult) pairs of the filtration
      :math:`(\ell, a_i)`, where the tree relation
      :math:`(\ell, a_i) \to (\ell+1, a_k)` holds if :math:`a_k` was a child of
      :math:`a_i` at level :math:`\ell`.  We say :math:`a_k` is a
      **successor** of :math:`a_i`, and :math:`a_i` is a **predecessor** of 
      :math:`a_k`.  Note that :math:`(\ell, a_i) \to (\ell+1, a_i)` for all
      :math:`a_i \in CL_\ell`

    Extending the maturation/reproduction metaphor of **adults**, **children**, and
    **guardians** above, a child :math:`x` with guardian :math:`a_i` at level
    :math:`\ell` is called a **teen** if :math:`\frac12 r_\ell < \|a_i - x\|`, and
    it is called a **youngin** if :math:`\|a_i - x\| \leq \frac12 r_\ell`.
    The point of this is that we may require the additional condition:
    
    - (Optional) On the previous condition, we can additionally require that
      each :math:`x` is the child of the *nearest* adult, if it lies in the
      intersection of two or more balls of :math:`B(a_i, r_\ell)`.  If two
      adults are equally distant, choose the one of the lowest index.  This
      option is enforced by the :code:`exchange_teens` flag.


    When changing from level math:`\ell` to level :math:`\ell+1`, the radius of
    each ball shrinks to :math:`r_{\ell+1} = \theta r_\ell`.  Children farther than
    :math:`r_{\ell+1}` from their guardians become **orphans**.  We must decide
    whether these orphans should be  **adopted** by other adults at level
    :math:`\ell+1`, or if the orphans should be **emancipated** as new adults at level
    :math:`\ell+1`.  That is, the newly emancipated adults at level
    :math:`\ell+1` comprise the **cohort** (see :code:`cohort`) at level $\ell+1$.

    We say :math:`a_j \in CL_{\ell}` is an **elder** of 
    :math:`a_k \in CL_{\ell+1}` if the distance :math:`\|a_j - a_k\|` is
    sufficiently small that :math:`a_j` *could have been* emancipated from
    :math:`a_k` between levels :math:`\ell` and :math:`\ell+1`.  That is,
    if the tree structure were unknown, then elders of :math:`a_j$ are the
    possible predecessors.  If :math:`a_k` is its own predecessor (because it
    was already an adult in :math:`CL_{\ell}`, then the only elder of
    :math:`a_k` is itself.

    **Example**

    Consider this point cloud in :math:`\mathbb{R}^2`

    .. math::

        X = \{(0,0.1),(1,2),(0,1),(0,0),(2,2),(2,2.2),(3,3),(1,1)\}

    We index these points from 0 to 7 in the given order.
    We have the following filtration

    .. math::

        &CL_0 = \{7\}\\
        &CL_1 = \{3, 4, 6, 7\}\\
        &CL_2 = \{1, 2, 3, 4, 6, 7\}\\
        &CL_3 = \{1, 2, 3, 4, 6, 7\}\\
        &CL_4 = \{1, 2, 3, 4, 5, 6, 7\}\\
        &CL_5 = \{0, 1, 2, 3, 4, 5, 6, 7\}\\
    
    We have the following cover ball radii

    .. math::

        &r_0 = 2\sqrt{2}\\
        &r_1 = \sqrt{2}\\
        &r_2 = \frac{\sqrt{2}}{2}\\
        &r_3 = \frac{\sqrt{2}}{4}\\
        &r_4= \frac{\sqrt{2}}{8}\\
        &r_5 = \frac{\sqrt{2}}{16}

    Here we have :math:`a_0 = (1,1)`, :math:`r_0 = 2\sqrt{2}`, and
    :math:`\theta = 1/2`.

    **The Friends Algorithm**

    Our algorithm is based upon the concept of **friends**. To each adult there
    will be associated *three* types of friends.  Types 1, 2,
    and 3 are used to build the `CoverTree` in typically linear time.

    Let :math:`a_i \in CL_\ell`, that is, :math:`a_i` is an adult at level
    :math:`\ell`. Define the following thresholds

    .. math::
        T_1(\ell) &= (2 + \theta)r_l \\
        T_2(\ell) &= (2 + 2\theta)r_l \\
        T_3(\ell) &= \frac{2}{1 - \theta}r_l.

    It is elementary to show that :math:`T_1(l) < T_2(l) < T_3(l)`.
    Moreover, we have the recursion relation
    :math:`T_3(l) < T_3(l-1)`. 
  
    Each level of the filtreation and all of this associated data is stored in 
    a `CoverLevel` object.

    The algorithm works like this, using a "reproduction" metaphor:

        - Level 0 (see :code:`covertree[0]` of type `CoverLevel`) has a single adult.  All
          points are its children.  Its only friends are itself.
        
        - ...
        
        - Level :math:`\ell` (see :code:`covertree[l]` of type `CoverLevel`)
          has known adults, friends1, friends3, friends3, and children.  We now
          compute level :math:`\ell+1.` in :func:`__next__`
          
          1. Shrink the radius by a factor of :math:`\theta`. Some children
          become orphans.
          
          2. Orphans are adopted or become newly emanicpated adults.  This uses
          :math:`T_1(\ell)`.
          
          3. If :code:`exhange_teens is True`, then children who are teens are
          re-assigned to the closest possible adult.  This uses
          :math:`T_2(\ell)`.
          
          4. Compute new friends3. 
          
          5. Use new friends3 to compute new friends1, friends2, friends3.
          
        - Level :math:`\ell+1` (see :code:`covertree[l+1]` of type `CoverLevel`)
          has known adults, friends1, friends3, friends3, and
          children.  We now compute level :math:`l+2`
        
        - ...

        - Stop when all points are adults.

    Levels are evaluated lazily and cached.  For example,
    if no levels have been computed, then
    :code:`covertree[3]` will compute levels 0, 1, 2, and 3.
    Then :code:`covertree[5]` will use those values for 0, 1, 2, 3 to compute 4
    and 5.

    Examples
    --------
    
    >>> pc = PointCloud.from_multisample_multilabel(
    ...     [np.array([[0,0.1],[1,2],[0,1],[0,0],[2,2],[2,2.2],[3,3],[1,1]])], [None])
    >>> ct = CoverTree(pc, ratio=0.5, sort_orphans_by_mean=False)
    >>> cl=ct.next()
    >>> list(cl.adults)
    [7]
    >>> pc.coords.values[7,:]
    array([ 1.,  1.])
    >>> cl
    Level 0 using 1 adults at radius 2.8284271247...
    >>> ct.next()
    Level 1 using 2 adults at radius 1.4142135623...
    >>> for cl in ct:
    ...     print(cl.exponent, list(cl.adults))
    0 [7]
    1 [7, 5]
    2 [7, 5, 0, 1, 2, 6]
    3 [7, 5, 0, 1, 2, 6]
    4 [7, 5, 0, 1, 2, 6, 4]
    5 [7, 5, 0, 1, 2, 6, 4, 3]
    >>> ct.cohort
    array([2, 2, 2, 5, 4, 1, 2, 0])

    References
    ----------
    .. [CDER1] Supervised Learning of Labeled Pointcloud Differences via Cover-Tree Entropy Reduction https://arxiv.org/abs/1702.07959
    .. [CDER2] CDER, Learning with Friends https://www.ima.umn.edu/2016-2017/DSS9.6.16-5.30.17/26150
    .. [iter1] https://docs.python.org/3/library/stdtypes.html?highlight=iterator#iterator-types
    .. [iter2] https://wiki.python.org/moin/Iterator
    """


    def __init__(self, pointcloud, ratio=ratio_Ag, exchange_teens=True, 
                 sort_orphans_by_mean=True):

        self.pointcloud = pointcloud
        self.pointcloud.covertree = self

        if np.any(self.pointcloud.stratum[0]['mass'].values <= 0):
            logging.warning("""
Some of your points have non-positive mass!  This is probably wrong.
Consider setting masses with PointCloud.stratum[0]['mass']=1.0.""")

        self.label_set = self.pointcloud.label_info['int_index'].values
        self.coords = self.pointcloud.coords.values
        try:
            self.pointcloud.multiplicity
        except AttributeError:
            self.pointcloud.multiplicity = np.ones(
                shape=(self.coords.shape[0],),
                dtype=np.int64)

        self.ratio = ratio
        self._levels = dict()
        self.radius = np.inf

        self.exchange_teens = exchange_teens
        self.sort_orphans_by_mean = sort_orphans_by_mean

        # more initialization happens in __next__()
        ball = self.pointcloud.cover_ball()
        self._r0 = ball['radius']
        self._adult0 = ball['index']
        
        self.N = self.pointcloud.coords.index.shape[0]
        self.allpoints = self.pointcloud.coords.index.values
        self.cohort = -1*np.ones(shape=(self.N,), dtype=np.int64)
        assert np.all(self.pointcloud.coords.index.values == np.arange(self.N)),\
            "So far, out methods require the pointcloud index to be range(N)."
        level0 = CoverLevel(self, 0)
        level0.adults.append(self._adult0)
        # TODO!  Use index method somehow!
        level0.children[self._adult0] = self.pointcloud.coords.index.values.copy()
        level0.friends1[self._adult0] = [self._adult0]
        level0.friends2[self._adult0] = [self._adult0]
        level0.friends3[self._adult0] = [self._adult0]
        level0.weights[self._adult0] = level0.find_label_weights(self._adult0)
        level0.predecessor = OrderedDict({self._adult0: None})
        level0.successors = OrderedDict()
        level0.guardians = self._adult0*np.ones(shape=(self.N,), dtype=np.int64)
        self.cohort[self._adult0] = 0
        
        self._levels[0] = level0
        self.level_pointer = -1
        self.reset()

    def __sizeof__(self):
        return sum( [cl.__sizeof__() for _,cl in self._levels.items()] )
            
    def __repr__(self):
        s = """A CoverTree of {} points in dimension {}, computed to
level\tadults\n""".format(
            self.pointcloud.coords.shape[0], 
            self.pointcloud.coords.shape[1])
        for cl in list(sorted(self._levels.keys())):
            s += "{}\t{}\n".format(cl, len(self._levels[cl].adults))
        return s

    def next(self):
        r""" See :func:`__next__` """
        return self.__next__()

    def __next__(self):
        r"""
        Increment exponent and compute/retrieve next level of cover tree as 
        a CoverLevel object.
        
        This is where the Friends algorithm happens.
        """
        assert 0.0 < self.ratio < 1.0
        
        # negative exponent means we are about to begin, so the next will be 0
        if self.level_pointer < 0:
            self.level_pointer = -1
        self.level_pointer += 1
        # simple cache
        if self.level_pointer in self._levels:
            return self._levels[self.level_pointer]
        
        assert self.level_pointer > 0

        # If we got here, we are really initialized.
        level = CoverLevel(self, self.level_pointer)

        # get data from previous level
        prev_level = self._levels[level.exponent - 1]
        
        # STEP 1: Promote
        level.guardians = deepcopy(prev_level.guardians)
        level.children = deepcopy(prev_level.children)
        level.adults = []
        level.adults.extend(prev_level.adults)
        for ca in level.adults:
            ci = ca  # for human sanity
            level.predecessor[ca] = ci
            prev_level.successors[ci] = np.array([ca], dtype=np.int64)
            # initialize friends -- updated cleverly later.
            level.friends1[ca] = [ca]
            level.friends2[ca] = [ca]
            level.friends3[ca] = [ca]
   
        # STEP 2: Orphan
        orphans = []
        for ci in level.adults:
            center_a = np.array([ci], dtype=np.int64)
            #children_ids = np.where(level.children[ci])[0]
            children_dists = fast_algorithms.distance_cache_None(center_a, level.children[ci], self.coords).flatten()
            # since we have computed children_dists, let's take a moment to count
            # duplicate points of new adults.
            if self.cohort[ci] == prev_level.exponent:
                mult = np.count_nonzero(children_dists == 0.0)
                self.pointcloud.multiplicity[ci] = mult
                if mult > 1:
                    logging.warning("point {} has multiplicity {}.".format(ci, mult))

            my_orphans = level.children[ci][children_dists > level.radius]
            assert np.all(np.in1d(my_orphans, level.children[ci]))

            if len(my_orphans) > 0 and self.sort_orphans_by_mean:
                child_coords = self.coords[level.children[ci], :]
                child_labels = self.pointcloud.labels[level.children[ci]]
                child_weight = self.pointcloud.stratum[0]['mass'].values[level.children[ci]]
                label_means, label_weights = fast_algorithms.label_means(
                                                child_coords,
                                                child_labels,
                                                child_weight,
                                                self.label_set)
                
                label_ordering = label_weights.argsort()[::-1]  # big-to-small
                dist_to_labelmean_by_orphan = cdist(label_means[label_ordering, :],
                                                    self.coords[my_orphans, :])
                # get closet-to-each-label until all orphans are used
                orphan_order = np.concatenate([
                    my_orphans[dist_to_labelmean_by_orphan.argsort(axis=1).T.flatten()], 
                    my_orphans])  # include everyone.

                # remove duplicates
                sort_orphan, sort_index = np.unique(orphan_order, return_index=True)
                assert len(my_orphans) == len(sort_index), "Orphans lost from sorted list?"
                # re-sort orphans by proximity to biggest weight. 
                # Because label_means was pre-sorted by weight, we can re-sort
                # by that index! 
                sort_index.sort()
                my_orphans = orphan_order[sort_index]
            orphans.extend(my_orphans)
        # check that each orphan was ejected once only.
        assert len(orphans) == len(set(orphans)), orphans
        # orphans = sorted(orphans)

        # STEP 3: Adopt or Liberate
        # Use type-1 friends to re-assign or promote orphans.
        # This is where most distances are computed, so it is the slowest.
        for orphan_index in orphans:
            assert orphan_index not in level.adults
            assert orphan_index in level.children[level.guardians[orphan_index]], "{} not in {}".format(orphan_index, level.children[level.guardians[orphan_index]])
            old_parent, new_parent = fast_algorithms.covertree_adopt_or_liberate(
                                    level, prev_level, orphan_index)
            if new_parent == orphan_index:
                prev_level.successors[old_parent] = np.append(prev_level.successors[old_parent], orphan_index)
                level.predecessor[orphan_index] = old_parent
                level.adults.append(orphan_index)
                level.guardians[orphan_index] = orphan_index
                level.children[orphan_index] = np.array([orphan_index], dtype=np.int64)
                level.friends1[orphan_index] = [orphan_index]
                level.friends2[orphan_index] = [orphan_index]
                level.friends3[orphan_index] = [orphan_index]
                self.cohort[orphan_index] = level.exponent
            assert orphan_index not in level.children[old_parent]
            assert orphan_index in level.children[new_parent]

        assert np.all(level.guardians >= 0)

        # STEP 4: Exchange teens
        # re-assign "teen" children to nearest adult using type-2 friends
        if self.exchange_teens:
            for ci in level.adults:
                fast_algorithms.covertree_exchange_teens(level, prev_level, ci)
        
        # STEP N:  Update friends from old friends
        prev_level = self._levels[level.exponent - 1]
        for pre_i in prev_level.adults:
            fast_algorithms.covertree_befriend321(level, prev_level, pre_i,
                                                  np.array(prev_level.friends3[pre_i], dtype=np.int64))
        
        level.cleanup()
        
        # assert level.check()
        self._levels[level.exponent] = level
        return level

    def reset(self):
        """
        Go to level -1.  Used internally to re-compute levels.
        """
        self.level_pointer = -1
        pass

    def __getitem__(self, exponent):
        """
        Get CoverLevel (exponent index, or slice of them) 
        """
        if isinstance(exponent, slice):
            # Since self[i] is already recursive, this probably makes 
            # a lot of excessive function calls, but oh well...
            return (self[i] for i in range(exponent.start, exponent.stop, exponent.step))
               
        else:
            if exponent < 0:
                exponent += len(self)

            self.reset()
            # ensure that previous levels have been computed
            while self.level_pointer < exponent:
                self.__next__()
            assert exponent == self.level_pointer
            return self._levels[exponent]

    def __iter__(self):
        r""" Iterate until stop condition is met or we run out of points. """
        self.reset()
        level = self.__next__()
        yield level

        num_points = self.pointcloud.coords.values.shape[0]
        while np.sum(self.pointcloud.multiplicity[level.adults]) < num_points:
            level = self.__next__()
            yield level

    def __len__(self):
        r"""Current Depth of the CoverTree.

        That is, the number of levels computed *so far*.  That is, if levels 
        0, 1, 2, 3 have been computed, then len(self) is 4. 

        Returns
        -------
        int

        """ 
        return max(self._levels.keys())+1

    def sparse_complex(self, level=-1):
        r""" Make a sparse complex from this CoverTree, using the type-4 friends 
        algorithm. 
      
        Notes
        -----
        This is a *placeholder*.  Sparse Complexes are not currently
        implemented in the stable codebase. 

        Parameters
        ----------
        level: int
            Level to use.  (Default: -1, meaning len(self)
            
        Returns
        -------
        PointCloud object, with edge values coming from sparse complex algorithm.
        """
        raise NotImplementedError

    def make_edges(self, min_distance=0.0, max_distance=-1.0):
        r"""Iterate over the edges between the points of the underlying
        `PointCloud`, where min_distance < length <= max_distance.

        Uses the CoverTree type-1 friends for efficiency.
        This is called by :func:`PointCloud.build_edges`

        Parameters
        ----------
        min_distance: float
            Minimum length.  (Default: 0.0)  Inequality means no self-edges!
        max_distance: float
            Maximum length.  (Default: -1.0, meaning 2*self._r0, for all edges)

        Yields
        ------
        triples (a,b,r), where a,b are the indices of points, and r is the
        distance.
        """

        if max_distance == -1.0:
            max_distance = 2*self._r0

        if max_distance <= 0.0:
            raise ValueError("Meaningless maximum distance {}.".format(max_distance))

        ell = np.int64(np.floor(np.log(max_distance/self._r0)/np.log(self.ratio)))
        ball_radius = self._r0 * (self.ratio ** ell)
        if ell <= 0:
            ell = 1
        else:
            assert ball_radius * self.ratio < max_distance <= ball_radius,\
                "Incorrect exponent?"
        
        # we need only check friends at level ell-1.
        level = self[ell - 1]
        total = 0
        for ci in level.adults:
            for cj in level.friends1[ci]:
                if ci == cj:
                    kids_i = level.children[ci]
                    total += int(len(kids_i)*(len(kids_i)-1)/2)
                    if len(kids_i) > 1:
                        dists = squareform(pdist(self.coords[kids_i,:], self.pointcloud.dist))
                        good_pairs = (min_distance < dists) & (dists <= max_distance)
                        good_edges = np.where(good_pairs)
                        for index_i, index_j in np.array(good_edges).T:
                            # don't double_count on symmetric square matrix!
                            if index_i < index_j:
                                yield (kids_i[index_i], kids_i[index_j], dists[index_i,index_j])

                # friends is reflexive, so don't double-count by parent
                elif ci < cj:
                    kids_i = level.children[ci]
                    kids_j = level.children[cj]
                    total += len(kids_i)*len(kids_j)
                    dists = fast_algorithms.distance_cache_None(kids_i,
                                                                kids_j,
                                                                self.coords)
                    good_pairs = (min_distance < dists) & (dists <= max_distance)
                    good_edges = np.where(good_pairs)
                    for index_i, index_j in np.array(good_edges).T:
                        yield (kids_i[index_i], kids_j[index_j], dists[index_i,index_j])
        if total > 0:
            logging.info("Examined {} possible edge distances using level {}.".format(total, ell-1))

    def plot(self, canvas, **kwargs):
        r""" Interactive plot of a CoverTree, with dynamic computation of levels.
        
        
        Parameters
        ----------
        canvas : :class:`bokeh.plotting.figure.Figure`
            as obtained from :code:`canvas = bokeh.plotting.figure()`
        
        Other parameters are fed to :func:`CoverLevel.plot`
        """
        if type(canvas).__module__ == 'bokeh.plotting.figure':
            canvas_type = "bokeh"
            import bokeh.plotting
            from bokeh.io import push_notebook
        # elif type(canvas).__module__ == 'matplotlib.axes._subplots':
        #    canvas_type = "pyplot"
        #    import matplotlib.pyplot as plt
        else:
            raise NotImplementedError(
                    """canvas must be a bokeh.plotting.figure() or a matplotlib.pyplot.subplots()[1].
                    You gave me {}""".format(type(canvas))
            )

        source = self[0].plot(canvas, **kwargs)

        def update(level):
            print("level {}".format(level))
            data, title = self[level].plot_data_title(**kwargs)
            canvas.title.text = title
            source.data = data
            push_notebook()
            pass
 
        return update
        # from ipywidgets import interact
        # return interact(update, level=(0,max(self._levels.keys())))

    def plot_tree(self, canvas, show_balls=True, show_tribes=False,
                  show_villages=False, show_adults=True):
        r""" Plot the tree of a CoverTree.
        
        Parameters
        ----------
        canvas : :class:`bokeh.plotting.figure.Figure`
            as obtained from :code:`canvas = bokeh.plotting.figure()`
        
        show_balls : boolean
            default True

        show_adults : boolean
            default True

        show_villages : boolean
            default False

        show_tribes : boolean
            default False

        """
        
        if type(canvas).__module__ == 'bokeh.plotting.figure':
            canvas_type = "bokeh"
            import bokeh.plotting
            from bokeh.io import push_notebook
        elif type(canvas).__module__ == 'matplotlib.axes._subplots':
            canvas_type = "pyplot"
            import matplotlib.pyplot as plt
        else:
            raise NotImplementedError(
                    "canvas must be a bokeh.plotting.figure().  You gave me {}".format(
                        type(canvas))
            )

        import networkx as nx

        g = nx.DiGraph()

        edges = []
        for root in self.tree:
            if root is not None:
                for branch in self.tree[root]:
                    edges.append((root, branch))
        g.add_edges_from(edges)
        
        val_map = dict((i, 1.0*self.cohort[i]/self.cohort.max()) for i in range(self.cohort.shape[0]))
        values = [val_map.get(node) for node in g.nodes()]
        
        pos = dict()
        prev_num = 0
        for ht in range(len(self)):
            adults = np.where(self.cohort == ht)[0]
            this_num = adults.shape[0]
            diff = this_num - prev_num
            prev_num = this_num
    
            for i,ci in enumerate(adults):
                pos[ci] = np.array([this_num/2.0 - i, 1.0*ht])

        for node in g.nodes():
            assert node in pos, "{} not found  {}". format(node, self.cohort[node])

        nx.draw_networkx_edges(g, pos, arrows=True, alpha=0.1)
        nx.draw_networkx_nodes(g, pos, node_size=50, cmap=plt.get_cmap('jet'), node_color = values)
        pass


class CoverLevel(object):
    r"""
    A thin class to represent one level of the filtration in a :class:`CoverTree`.
    A CoverLevel is essentially a collection of dictionaries of adults,
    friends, children, and other attributes of a particular level.
    
    The various attributes have different orderings, optimized for typical
    usage and minimal algorithmic complexity.

    Notes
    -----
    The user should never create a CoverLevel directly.  Instead, create a
    CoverTree and access its :math:`i^{\text{th}}` level with 
    :code:`covertree[i]`.


    Attributes
    ----------
    covertree : :class:`CoverTree`
        The CoverTree to which this CoverLevel belongs.
    pointcloud : :class:`multidim.PointCloud`
        The PointCloud used to make the CoverTree 
    exponent : int
        The exponent (that is, index or depth or level) of this CoverLevel in
        the CoverTree.
    radius : :class:`numpy.float64`
        The ball radius
    T1 : :class:`numpy.float64`
        The type-1 friends radius
    T2 : :class:`numpy.float64`
        The type-2 friends radius
    T3 : :class:`numpy.float64`
        The type-3 friends radius
    adults : `list`
        List of adult indices, in order they were born
    friends1 : :class:`collections.OrderedDict`
        An ordered dictionary to keep track of type-1 friends. Keyed by the
        adults, in birth order.  The values are lists, in index order.
    friends2 : :class:`collections.OrderedDict`
        An ordered dictionary to keep track of type-2 friends. Keyed by the
        adults, in birth order.  The values are lists, in index order.
    friends3 : :class:`collections.OrderedDict`
        An ordered dictionary to keep track of type-3 friends. Keyed by the
        adults, in birth order. The values are lists, in index order.
    guardians : :class:`numpy.ndarray` 
        An array of :class:`numpy.int64`, which keeps track of the guardians
        of each point in the underlying `PointCloud`.  Adults are their own
        guardians.
    predecessor : :class:`collections.OrderedDict`
        An ordered dictionary to keep track of predecessors of the adults.
        Keyed by the adults, in birth order.  The values are
        the indices of adults at the previous `CoverLevel`.
    successors : :class:`collections.OrderedDict`
        An ordered dictionary to keep track of predecessors of the adults.
        Keyed by the adults, in birth order.  The values are
        NumPy arrays of indices of adults in the next `CoverLevel`.  This is
        computed only at the next level!
    children : :class:`collections.OrderedDict`
        An ordered dictionary to keep track of children1 friends, keyed by the
        adults, in birth order.  The values are NumPy boolean arrays, which
        allows for easy extraction of subsets of children. 
    weights : :class:`collections.OrderedDict`
        An ordered dictionary to keep track of total weight of children, keyed 
        by the adults, in birth order.  The values are NumPy arrays, with one
        entry per label.  This is computed as part of :func:`cleanup`
    entropy : :class:`collections.OrderedDict`
        An ordered dictionary to keep track of overall entropy of children, keyed
        by the adults, in birth order.  The values are :class:`numpy.float64` 
        numbers, of overall entropy of weights across labels.  This is computed
        and stored via :class:`multidim.models.CDER`, but it otherwise
        unused.
    """
    def __init__(self, covertree, exponent):
        self.covertree = covertree
        self.pointcloud = self.covertree.pointcloud
        self.exponent = exponent
        
        self.radius = self.covertree._r0 * (self.covertree.ratio ** self.exponent)
        self.T1 = self.radius*(2.0 + self.covertree.ratio)
        self.T2 = self.radius*(2.0 + 2.0*self.covertree.ratio) 
        self.T3 = self.radius*2.0/(1.0 - self.covertree.ratio)
       
        self.adults = []
        self.friends1 = OrderedDict()  # each entry should be a LIST
        self.friends2 = OrderedDict()  # each entry should be a LIST
        self.friends3 = OrderedDict()  # each entry should be a LIST

        self.guardians = None
        self.predecessor = OrderedDict()  # each entry an ARRAY
        self.successors = OrderedDict()   # each entry an ARRAY
        self.children = OrderedDict()  # each entry a np.uint8 ARRAY
        self.weights = OrderedDict()  # each entry a np array by label
        self.entropy = OrderedDict()  # each entry a np.float64

    def check(self):
        r""" Perform basic sanity checks on children, friends, etc. 
        Throws `AssertionError` if anyhting fails.
        """
        
        assert type(self.adults) == list
        assert type(self.friends1) == OrderedDict
        assert type(self.friends2) == OrderedDict
        assert type(self.friends3) == OrderedDict
        assert type(self.predecessor) == OrderedDict
        assert type(self.successors) == OrderedDict
        assert type(self.weights) == OrderedDict
        assert type(self.entropy) == OrderedDict

        assert type(self.guardians) == np.ndarray\
            and self.guardians.shape == (self.covertree.N, )

        adult_set = set(self.adults)
        assert len(adult_set) == len(self.adults)

        assert set(self.children.keys()) == adult_set, "Mismatched adults and children keys"
        assert set(self.friends1.keys()) == adult_set, "Mismatched adults and friends1 keys"
        assert set(self.friends2.keys()) == adult_set, "Mismatched adults and friends2 keys"
        assert set(self.friends3.keys()) == adult_set, "Mismatched adults and friends3 keys"
        assert set(self.predecessor.keys()) == adult_set, "Mismatched adults and predecessor keys"
        assert set(self.successors.keys()) == set() or set(self.successors.keys()) == adult_set,\
            "Mismatched adults and successors keys"
        assert set(self.weights.keys()) == set() or set(self.weights.keys()) == adult_set,\
            "Mismatched adults and weights keys"
        assert set(self.entropy.keys()) == set() or set(self.entropy.keys()) == adult_set,\
            "Mismatched adults and entropy keys"

        assert set(list(self.guardians)) == adult_set, "Mismatched guardians and adults."

        # cannot check successors without violating something..

        union = np.array([], dtype=np.int64)
        for ci in self.adults:
            assert type(self.friends1[ci]) == list
            assert type(self.friends2[ci]) == list
            assert type(self.friends3[ci]) == list
            assert type(self.children[ci]) == np.ndarray\
                and self.children[ci].dtype == 'int64'\
                and self.children[ci].shape[0] <= self.covertree.N

            assert len(set(self.friends1[ci])) == len(self.friends1[ci])
            assert len(set(self.friends2[ci])) == len(self.friends2[ci])
            assert len(set(self.friends3[ci])) == len(self.friends3[ci])

            assert ci in self.friends1[ci]
            assert ci in self.friends2[ci]
            assert ci in self.friends3[ci]
            assert self.guardians[ci] == ci
            assert ci in self.children[ci]

            assert np.intersect1d(union, self.children[ci]).shape[0] == 0,\
                "Children overlap.  Not Partition."

            union = np.union1d(union, self.children[ci])
        assert len(union) == self.covertree.N, "Children missing.  Not Partition."
        #assert len(union) == len(np.unique(union)), "Duplicates?"

        try:
            v = self.villages
            # TODO -- also test matching indices
            assert len(v._blocks) == len(self.adults),\
                "blocks should match adults"
        except AttributeError:
            pass

        return True

    def __sizeof__(self):
        import sys
        return  sum([sys.getsizeof(x) for x in [
            self.adults,
            self.children,
            self.friends1,
            self.friends2,
            self.friends3,
            self.guardians,
            self.predecessor,
            self.weights,
            self.entropy]])


    def __repr__(self):
        return "Level {} using {} adults at radius {}".format(
            self.exponent, len(self.adults), self.radius)

    def find_label_weights(self, adult):
        r""" Compute the weights of labelled children of an adult. 
        Store it in self.weights[adult].

        Parameters
        ----------
        adult : `int`
            index of adult to compute.

        Returns 
        -------
        self.weights[adult]

        """
        if adult in self.weights.keys():
            pass
        else:
            pc = self.covertree.pointcloud
            children_set = np.zeros(shape=(pc.coords.shape[0],), dtype='bool')
            children_set[self.children[adult]] = True
            self.weights[adult] = fast_algorithms.label_weights(
                children_set,
                pc.labels,
                pc.stratum[0]['mass'].values,
                pc.label_info['int_index'].values)
        return self.weights[adult]

    def find_entropy(self, adult):
        r""" Compute the entropy of the labelled children on an adult. 
        Store it in self.entropy[adults].

        This is only used by :class:`multidim.models.CDER`

        Parameters
        ----------
        adult : `int`
            index of adult to compute.

        Returns 
        -------
        self.entropy[adult]

        """
        if adult in self.entropy.keys():
            pass
        else:
            totweight = self.weights[adult].sum()
            assert totweight > 0
            self.entropy[adult] = fast_algorithms.entropy(self.weights[adult]/totweight)
        return self.entropy[adult]


    def plot_data_title(self, show_balls=True, show_adults=True):
        r""" Internal method -- Make source data for plot. 
        
        See Also
        --------
        :func:`plot`
        
        """

        title = "CoverTree Level {}, radius {}".format(self.exponent, self.radius)
        pc = self.covertree.pointcloud

        xts = []
        cts = []
        import bokeh.palettes
            
        adult_ids = sorted(list(self.adults))

        xs = pc.coords.loc[adult_ids, 0].values
        ys = pc.coords.loc[adult_ids, 1].values
        rs = [self.radius]*len(self.adults)
        cs = [str(c) for c in self.adults]
        data = {'xs': xs, 'ys': ys, 'rs': rs, 'cs': cs, 'cts': xts, 'cts': cts}

        return data, title

    def cleanup(self):
        r""" Internal method -- remove duplicate friends, and compute weights
        and entropy.
        """
        for ca in self.adults:
            self.friends1[ca] = sorted(list(set(self.friends1[ca])))
            self.friends2[ca] = sorted(list(set(self.friends2[ca])))
            self.friends3[ca] = sorted(list(set(self.friends3[ca])))
            self.find_label_weights(ca)
 
    def plot(self, canvas, show_balls=True, show_adults=True, show_hulls=False, color='purple'):
        """
        Plot a single level of a `CoverTree`
        
        See the example at example-covertree_ 

        Parameters
        ----------
        canvas : :class:`bokeh.plotting.figure.Figure` 
            as obtained from :code:`canvas = bokeh.plotting.figure()`
        show_balls : bool
            Draw the covering balls at this level.
            Default: True
        show_adults : bool
            Draw the adults at this level.
            Default: True
        show_hulls : bool
            Draw the convex hulls of the childeren of each adult.
            Note -- this works only with matplotlib for now, not bokeh.
            Default: False
        color : str
            Name of color to use for cover-tree balls and hulls.
            Default: 'purple'

        References  
        ----------
        .. _example-covertree: http://nbviewer.jupyter.org/github/geomdata/gda-public/blob/master/examples/example-covertree.ipynb

        """

        # fix the aspect ratio!
        all_xs = self.pointcloud.coords.values[:, 0]
        all_ys = self.pointcloud.coords.values[:, 1]
        xmid = (all_xs.max() + all_xs.min())/2.0
        ymid = (all_ys.max() + all_ys.min())/2.0
        span = max([all_xs.max() - xmid,
                    xmid - all_xs.min(),
                    all_ys.max() - ymid,
                    ymid - all_ys.min()])

        if type(canvas).__module__ == 'bokeh.plotting.figure':
            canvas_type = "bokeh"
            from bokeh.models import ColumnDataSource, Range1d
            import bokeh.plotting
        elif type(canvas).__module__ == 'matplotlib.axes._subplots':
            canvas_type = "pyplot"
            import matplotlib.pyplot as plt
            from matplotlib.collections import PolyCollection, PatchCollection
            from matplotlib.patches import Circle, Ellipse, Polygon
            import matplotlib.colors as colors
             
            # fix the aspect ratio!
            canvas.set_aspect('equal')
            canvas.set_xlim([xmid-span, xmid+span])
            canvas.set_ylim([ymid-span, ymid+span])

        else:
            raise NotImplementedError(
                    "canvas must be a bokeh.plotting.figure().  You gave me {}".format(
                        type(canvas))
            )

        pc = self.covertree.pointcloud
        all_xs = pc.coords.values[:, 0]
        all_ys = pc.coords.values[:, 1]
        data, title = self.plot_data_title(show_balls=show_balls, 
                                           show_adults=show_adults)

        # fix the aspect ratio! 
        xmean = all_xs.mean()
        ymean = all_ys.mean()
        span = max([all_xs.max() - xmean,
                    xmean - all_xs.min(),
                    all_ys.max() - ymean,
                    ymean - all_ys.min()])

        if canvas_type == "pyplot":
            xs = data['xs']
            ys = data['ys']
            rs = data['rs']
            if show_balls:
                patches = []
                rgbas = []
                cc = colors.ColorConverter()
                for i in range(len(xs)):
                    patches.append(Circle(xy=(xs[i], ys[i]), radius=rs[i]))
                    # have to set the alpha value manually.
                    rgba = list(cc.to_rgba(color))
                    rgba[3] = 0.2
                    rgbas.append(tuple(rgba))
                p = PatchCollection(patches, edgecolor='none')
                p.set_facecolors(rgbas)
                canvas.add_collection(p)
            
            if show_adults:
                canvas.scatter(x=xs, y=ys, color='blue', alpha=1.)

            if show_hulls:
                from scipy.spatial import ConvexHull
                patches = []
                rgbas = []
                cc = colors.ColorConverter()
                for ai in self.adults:
                    children = pc.coords.values[self.children[ai], :]
                    if children.shape[0] >= 3:
                        hull = ConvexHull(children).vertices
                        poly_data = children[hull, :]
                        patches.append(Polygon(poly_data))
                    elif children.shape[0] == 2:
                        d = cdist( children[[0],:], children[[1],:] )[0,0]
                        patches.append(Circle(xy=pc.coords.values[ai,:], radius=d))
                    else: # singleton
                        patches.append(Circle(xy=pc.coords.values[ai,:], radius=0.5*self.radius))
                    rgba = list(cc.to_rgba(color))
                    rgba[3] = 0.2
                    rgbas.append(tuple(rgba))
 
                p = PatchCollection(patches, edgecolor=color)
                p.set_facecolors(rgbas)
                canvas.add_collection(p)
                pass


        elif canvas_type == "bokeh":
            source = ColumnDataSource(data=data)
            canvas.title.text = title
            canvas.x_range = Range1d(xmean-span, xmean+span)
            canvas.y_range = Range1d(ymean-span, ymean+span)
            if show_balls:
                canvas.circle('xs', 'ys', source=source, radius='rs', color=color, alpha=0.2)
   
            if show_adults:
                canvas.circle('xs', 'ys', source=source, size=4, color='blue', alpha=1.)
       
            if show_hulls:
                raise NotImplementedError("No hulls in Bokeh yet. Use pyplot.")
 
            #canvas.circle(all_xs, all_ys, color='black', alpha=0.2, size=0.5)

            return source
