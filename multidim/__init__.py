# coding=utf-8

r"""
The `multidim` class provides user-facing tools for topological data analysis
of multi-dimensional data.

The goal is to be honest about topology while also using speed/cleverness with
a minimal amount of user headache.

Included are:
    - `PointCloud`, for data points in Euclidean space.
    - `SimplicialComplex`, for abstract simplicial complexes, built
      from `Simplex` objects sorted by dimension into `SimplexStratum` objects.


Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE

Examples
--------

>>> X = np.load("tests/circle.npy")
>>> pc = PointCloud(X, max_length=-1)
>>> pc
A SimplicialComplex with 1000 points, 499500 edges, and 0 faces.
>>> np.all(pc.stratum[0]['pos'].values == True)
True
>>> pc.check()
>>> pc.make_pers0(cutoff=0.15)
>>> for v in pc.cells(0):
...     if v.positive:
...         print(v)
0+ Simplex 0 of height 0.0 and mass 1.0
0+ Simplex 74 of height 0.0 and mass 1.0
0+ Simplex 183 of height 0.0 and mass 1.0
0+ Simplex 195 of height 0.0 and mass 1.0
0+ Simplex 197 of height 0.0 and mass 1.0
0+ Simplex 231 of height 0.0 and mass 1.0
0+ Simplex 354 of height 0.0 and mass 1.0
0+ Simplex 397 of height 0.0 and mass 1.0
0+ Simplex 489 of height 0.0 and mass 1.0
0+ Simplex 530 of height 0.0 and mass 1.0
0+ Simplex 607 of height 0.0 and mass 1.0
0+ Simplex 757 of height 0.0 and mass 1.0
0+ Simplex 781 of height 0.0 and mass 1.0
0+ Simplex 800 of height 0.0 and mass 1.0
0+ Simplex 903 of height 0.0 and mass 1.0
0+ Simplex 980 of height 0.0 and mass 1.0
>>> pc.pers0.grab(5)['keepcode']
     birth_index  death_index  birth     death      pers
979          213          316    0.0  0.136923  0.136923
980          135          135    0.0  0.136992  0.136992
981          439          477    0.0  0.138059  0.138059
982          610          630    0.0  0.138474  0.138474
983          603          603    0.0  0.139332  0.139332
>>> pc.make_pers1_rca1(cutoff=0.2)
>>> pc.pers1.grab(5)['keepcode']
     birth_index  death_index     birth     death      pers
221         3217         9700  0.095619  0.168120  0.072501
220         2942         9661  0.091542  0.167720  0.076177
219         2713         9279  0.087438  0.164152  0.076713
224         3333        10439  0.097564  0.174643  0.077079
200         1816         7688  0.071490  0.149336  0.077846
>>> V=pc.stratum[0]
>>> V.loc[:10]
    height  mass    pos  rep
0      0.0   1.0   True    0
1      0.0   1.0  False    0
2      0.0   1.0  False    1
3      0.0   1.0  False    0
4      0.0   1.0  False    0
5      0.0   1.0  False    4
6      0.0   1.0  False    1
7      0.0   1.0  False    0
8      0.0   1.0  False    0
9      0.0   1.0  False    0
10     0.0   1.0  False    1
>>> pc.cells(0)[0]
0+ Simplex 0 of height 0.0 and mass 1.0
>>> pc.cells(0)[2]
0- Simplex 2 of height 0.0 and mass 1.0
>>> E=pc.stratum[1]
>>> E.loc[:10]
      height    pos  rep  bdy0  bdy1
0   0.001142  False    0   858   866
1   0.001997  False    1    98   187
2   0.002471  False    2   251   313
3   0.002670  False    3   599   629
4   0.002766  False    4   150   167
5   0.003405  False    5   573   620
6   0.003812  False    6   474   517
7   0.005357  False    7   893   988
8   0.005533  False    8   623   644
9   0.005914  False    9   648   744
10  0.006056  False   10   612   640
>>> pc.cells(1)[2]
1- Simplex 2 of height 0.0024707293775457456 and mass None
"""

from __future__ import print_function

import itertools
import numpy as np
import pandas as pd
# Don't roll our own L2 norms
from scipy.spatial.distance import squareform, cdist, pdist, is_valid_dm
from . import multidim_fast_algorithms
import homology.dim0
import homology.dim1


class Simplex(object):
    r"""
    This class is a convenient container to access the data in the
    pd DataFrame stratum[dim] of a SimplicialComplex.
    It is always faster to access stratum[dim].loc[index] directly.

    Parameters
    ----------
    cellcomplex : :class:`SimplicialComplex`
        The SimplicialComplex to which this Simplex belongs.
    dim : int
        The dimension in which this Simplex lives.
    index : int
        The abstract index of this Simplex.

    Attributes
    ----------
    cellcomplex : `SimplicialComplex`
        The `SimplicialComplex` to which this Simplex belongs.
    dim : int
        The dimension in which this Simplex lives.
    index : int
        The abstract index of this Simplex.
    shadow_complex : `SimplicialComplex`
    children : :class:`pandas.Series`

    See Also
    --------
    SimplicialComplex : A class for abstract simpicial cell complexes.

    Notes
    -----
    A finite (*abstract*) *simplicial complex* is a finite set :math:`A` together
    with collection :math:`\Delta` of subsets of :math:`A` such that if
    :math:`X \in \Delta` and :math:`Y \subset X` then
    :math:`Y \in \Delta`. [1]_

    References
    ----------
    .. [1] D. Feichtner-Kozlov, Combinatorial algebraic topology.
        Berlin: Springer, 2008.

    """

    def __init__(self, cellcomplex, dim, index):
        self.cellcomplex = cellcomplex
        self.dim = dim
        self.index = index
        self.shadow_complex = None
        self.children = pd.Series(dtype=np.float64)

    @property
    def height(self):
        r"""
        :return: height (that is, filtered value) of this cell (np.float64)
        """
        return self.cellcomplex.stratum[self.dim]['height'].loc[self.index]

    @height.setter
    def height(self, v):
        self.cellcomplex.stratum[self.dim]['height'].loc[self.index] = v

    @property
    def mass(self):
        r"""
        :return: mass of this cell (np.float64 or None)
        """
        if 'mass' in self.cellcomplex.stratum[self.dim]:
            return self.cellcomplex.stratum[self.dim]['mass'].loc[self.index]
        else:
            return None

    @mass.setter
    def mass(self, v):
        self.cellcomplex.stratum[self.dim]['mass'].loc[self.index] = v

    @property
    def positive(self):
        r"""

        :return:
        """
        return self.cellcomplex.stratum[self.dim]['pos'].loc[self.index]

    @positive.setter
    def positive(self, b):
        r"""

        :param b:
        """
        self.cellcomplex.stratum[self.dim]['pos'].loc[self.index] = b

    @property
    def representative(self):
        r"""

        :return:
        """
        return self.cellcomplex.stratum[self.dim]['rep'].loc[self.index]

    @representative.setter
    def representative(self, r):
        r"""

        :param r:
        """
        self.cellcomplex.stratum[self.dim]['rep'].loc[self.index] = r

    @property
    def boundary(self):
        r"""

        :return:
        """
        parts = []
        if self.dim > 0:
            parts = range(self.dim + 1)
        return {self.cellcomplex.stratum[self.dim]['bdy{}'.format(j)].loc[self.index] for j in parts}

    @boundary.setter
    def boundary(self, s):
        r"""

        :param s:
        """
        for i, c in enumerate(sorted(list(s))):
            self.cellcomplex.stratum[self.dim]['bdy{}'.format(i)].loc[
                self.index] = c

    def __hash__(self):
        r"""

        :param self:
        :return:
        """
        return self.index

    def __eq__(self, other):
        r"""

        :param other:
        :return:
        """
        return self.cellcomplex == other.cellcomplex and self.__hash__() == other.__hash__()

    def __repr__(self):
        r"""

        :return:
        """
        sign = "+"
        if not self.positive:
            sign = "-"
        return "{}{} Simplex {} of height {} and mass {}".format(self.dim, sign, self.index,
                                                    repr(self.height),
                                                    repr(self.mass))

    def __lt__(self, other):
        if not (self.cellcomplex == other.cellcomplex):
            raise ValueError("These Cells are not in the same SimplicialComplex!")
        if not (self.dim == other.dim):
            raise ValueError("These Cells are not of the same dimension!")
        return self.height() < other.height()


class SimplexStratum(object):
    r""" SimplexStratum is a thin class for calling :class:`Simplex` objects of a certain
    dimension from a `SimplicialComplex`.  It is an interface to the data in
    `SimplicialComplex.stratum`[dim], which is a `pandas.DataFrame`.  Whenever
    possible, the `pandas.DataFrame` should be called directly, for speed.
    """

    def __init__(self, cell_complex, dim):
        self.cell_complex = cell_complex
        self.dim = dim
        self._cells = dict()

    def __getitem__(self, i):
        if i not in self._cells:
            self._cells[i] = Simplex(self.cell_complex, self.dim, i)
        return self._cells[i]

    def __iter__(self):
        for i in self.cell_complex.stratum[self.dim].index:
            yield self[i]

    def __repr__(self):
        return "Stratum {} of SimplicialComplex {}".format(self.dim,
                                                           id(self.cell_complex))

    def __hash__(self):
        return id(self.cell_complex), self.dim

    def __eq__(self, other):
        return type(self) == other(self) and self.__hash__ == other.__hash__


def stratum_maker(dim=0):
    r"""
    Make an empty stratum :class:`pandas.DataFrame` of the appropriate dimension.
    This is used to initialize a new dimension of a :class:`SimplicialComplex`.

    Parameters
    ----------
    dim : int
        Dimension of stratum (0, 1, 2, ...)

    Returns
    -------
    DataFrame : :class:`pandas.DataFrame`
         pd DataFrame suitable for SimplicialComplex.stratum[dim]

    See Also
    --------
    :class:`SimplicialComplex` : A class for abstract simplicial cell complexes.
    """

    bdy_size = 0
    if dim > 0:
        bdy_size = dim + 1
    return pd.DataFrame({},
                        columns=['height', 'pos', 'rep'] + ['bdy{}'.format(i)
                                                         for i in
                                                         range(bdy_size)],
                        index=range(0))


def stratum_from_distances(dists, max_length=-1.0, points=None):
    r""" Construct a stratum dictionary from a symmetric matrix of distances.

    Parameters
    ----------
    dists : :class:`numpy.ndarray`
        A symmetric NxN array for distances, as obtained from
        :class:`scipy.spatial.distances.squareform`
    max_length : int
        If max_length >=0, store only those edges
        of length < max_length. Default: -1.0, store all edges.
    points : :class:`pandas.DataFrame`
        A fully-formed DataFrame of point information for stratum[0].

    Returns
    -------
    {0: points, 1: edges} : dict
        A stratum dictionary, suitable for SimplicialComplex objects.

    See Also
    --------
    :func:`fast_algorithms.edges_from_dists`
    """
    is_valid_dm(dists, throw=True)

    if points is None:
        n = dists.shape[0]
        idx0 = np.arange(n, dtype=np.int64)
        hgt0 = np.zeros(n, dtype=np.float64)
        pos0 = np.ones(shape=(n,), dtype='bool')
        points = pd.DataFrame({
            'height': hgt0,
            'pos': pos0,
            'rep': idx0,
        },
                columns=['height', 'pos', 'rep'],
                index=idx0)

    if max_length == 0:
        # if the cutoff is 0, we don't want to bother to make all the
        # distances.
        edges = stratum_maker(1)
    else:
        hgt1, pos1, bdys = fast_algorithms.edges_from_dists(points.index.values, dists,
                                                            cutoff=np.float64(max_length))
        num_edges = hgt1.shape[0]
        idx1 = np.arange(num_edges, dtype='int64')
        edges = pd.DataFrame({
            'height': hgt1,
            'pos': pos1,
            'rep': idx1,
            'bdy0': bdys[:, 0],
            'bdy1': bdys[:, 1],
        },
                columns=['height', 'pos', 'rep', 'bdy0', 'bdy1'],
                index=idx1)
    return {0: points, 1: edges}


def lower_star_for_image(img_array, diagonals_and_faces=True):
    """
    Compute the lower star weighted simplicial complex from a 2d grid/image.

    Parameters
    ----------
    img_array, a `numpy.ndarray` of dimension 2.

    Returns
    -------
    `homology.SimplicialComplex`

    Examples
    --------

    >>> A = np.random.rand(3,4)
    >>> lower_star_for_image(A)
    A SimplicialComplex with 12 points, 23 edges, and 12 faces.
    """
    assert len(img_array.shape) == 2,\
        "Lower-star filtration is currently for images (2d arrays) only."

    m = img_array.shape[0]
    n = img_array.shape[1]

    # make all vertices, by flattening val_array and indexing in the normal way
    verts_hgt = img_array.flatten()
    verts_rep = np.arange(m*n)
    flat_index = verts_rep.reshape(m, n)

    edges_hgt = []
    edges_rep = []
    edges_bdy0 = []
    edges_bdy1 = []
    # Make all the horizontal edges.
    for i, j in itertools.product(range(m), range(n-1)):
        # collect vertices' indices and heights
        # left=(i,j) -- right=(i,j+1)
        lf_idx = flat_index[i, j]
        rt_idx = flat_index[i, j+1]
        # There is no real reason for these asserts -- just clarification.
        # assert lf_idx == n*(i) + (j)
        # assert rt_idx == n*(i) + (j+1)

        lf_hgt = img_array[i, j]
        rt_hgt = img_array[i, j+1]
        # There is no real reason for these asserts -- just clarification.
        # assert lf_hgt == verts_hgt[lf_idx]
        # assert rt_hgt == verts_hgt[rt_idx]

        edges_hgt.append(np.max([lf_hgt, rt_hgt]))
        edges_rep.append(len(edges_rep))
        edges_bdy0.append(lf_idx)
        edges_bdy1.append(rt_idx)

        # This i,j horizontal edge should have index (n-1)*i + j
        assert len(edges_hgt) - 1 == (n-1)*i + j

    # did we count all horizontal edges?
    assert len(edges_hgt) == (n-1)*m

    # Make all the vertical edges
    for i, j in itertools.product(range(m-1), range(n)):
        # collect vertices' indices and heights
        # top=(i,j)
        #     |
        # bot=(i+1,j)
        tp_idx = flat_index[i, j]
        bt_idx = flat_index[i+1, j]
        # There is no real reason for these asserts -- just clarification.
        # assert tp_idx == n*(i) + (j)
        # assert bt_idx == n*(i+1) + (j)
        tp_hgt = img_array[i, j]
        bt_hgt = img_array[i+1, j]
        # There is no real reason for these asserts -- just clarification.
        # assert tp_hgt == verts_hgt[tp_idx]
        # assert bt_hgt == verts_hgt[bt_idx]
        edges_hgt.append(np.max([tp_hgt, bt_hgt]))
        edges_rep.append(len(edges_rep))
        edges_bdy0.append(tp_idx)
        edges_bdy1.append(bt_idx)

        # This i,j vertical edge should have index n*i + j
        # AFTER the (n-1)*m horizontal edges
        assert len(edges_hgt) - 1 == (n-1)*m + n*i + j

    # did we cound all vertical AND horizontal edges?
    assert len(edges_hgt) == (n-1)*m + n*(m-1)

    faces_hgt = []
    faces_rep = []
    faces_bdy0 = []
    faces_bdy1 = []
    faces_bdy2 = []

    if diagonals_and_faces:
        # Make the diagonal edges, and the faces, too.
        for i, j in itertools.product(range(m-1), range(n-1)):
            # collect the vertices' indices and heights
            #  nw=(i,j)        ne=(i, j+1)
            #          at (i,j)
            #  sw=(i+1, j)     se=(i+1, j+1)
            nw_idx = flat_index[i, j]
            ne_idx = flat_index[i, j+1]
            se_idx = flat_index[i+1, j+1]
            sw_idx = flat_index[i+1, j]
            # There is no real reason for these asserts -- just clarification.
            # assert nw_idx == n*(i) + (j)
            # assert ne_idx == n*(i) + (j+1)
            # assert se_idx == n*(i+1) + (j+1)
            # assert sw_idx == n*(i+1) + (j)
    
            nw_hgt = img_array[i, j]
            ne_hgt = img_array[i, j+1]
            se_hgt = img_array[i+1, j+1]
            sw_hgt = img_array[i+1, j]
            # There is no real reason for these asserts -- just clarification.
            # assert nw_hgt == verts_hgt[nw_idx]
            # assert ne_hgt == verts_hgt[ne_idx]
            # assert se_hgt == verts_hgt[se_idx]
            # assert sw_hgt == verts_hgt[sw_idx]
    
            # determine diagonal
            cell_max_loc = np.argmax([nw_hgt, ne_hgt, se_hgt, sw_hgt])
    
            if cell_max_loc % 2 == 0:
                # Max is either nw or se.
                # Make edge (nw,se)
                edges_hgt.append(np.max([nw_hgt, se_hgt]))
                edges_rep.append(len(edges_rep))
                edges_bdy0.append(nw_idx)
                edges_bdy1.append(se_idx)
    
                # Make face (nw,ne,se).
                faces_hgt.append(np.max([nw_hgt, ne_hgt, se_hgt]))
                faces_rep.append(len(faces_rep))
                faces_bdy0.append((n-1)*i + j)  # horizontal nw-ne
                # assert edges_bdy0[ (n-1)*i + j ] == nw_idx
                # assert edges_bdy1[ (n-1)*i + j ] == ne_idx
                faces_bdy1.append((n-1)*m + n*i + j+1)  # vertical ne|se
                # assert edges_bdy0[ (n-1)*m + n*i + j+1 ] == ne_idx
                # assert edges_bdy1[ (n-1)*m + n*i + j+1 ] == se_idx
                faces_bdy2.append(edges_rep[-1])  # most recent edge is nw\se
                # assert edges_bdy0[ edges_rep[-1] ] == nw_idx
                # assert edges_bdy1[ edges_rep[-1] ] == se_idx
    
                # Make face (sw,se,nw).
                faces_hgt.append(np.max([sw_hgt, se_hgt, nw_hgt]))
                faces_rep.append(len(faces_rep))
                faces_bdy0.append((n-1)*(i+1) + j)  # horizontal sw-se
                # assert edges_bdy0[ (n-1)*(i+1) + j ] == sw_idx
                # assert edges_bdy1[ (n-1)*(i+1) + j ] == se_idx
                faces_bdy1.append((n-1)*m + n*i + j)  # vertical nw|sw
                # assert edges_bdy0[ (n-1)*m + n*i + j ] == nw_idx
                # assert edges_bdy1[ (n-1)*m + n*i + j ] == sw_idx
                faces_bdy2.append(edges_rep[-1])  # most recent edge is nw\se
                # assert edges_bdy0[ edges_rep[-1] ] == nw_idx
                # assert edges_bdy1[ edges_rep[-1] ] == se_idx
    
            else:
                # Max is either ne or sw.
                # Make edge (ne,sw)
                edges_hgt.append(np.max([ne_hgt, sw_hgt]))
                edges_rep.append(len(edges_rep))
                edges_bdy0.append(ne_idx)
                edges_bdy1.append(sw_idx)
    
                # Make face (nw,ne,sw).
                faces_hgt.append(np.max([nw_hgt, ne_hgt, sw_hgt]))
                faces_rep.append(len(faces_rep))
                faces_bdy0.append((n-1)*i + j)  # horizontal nw-ne
                # assert edges_bdy0[ (n-1)*i + j ] == nw_idx
                # assert edges_bdy1[ (n-1)*i + j ] == ne_idx
                faces_bdy1.append((n-1)*m + n*i + j)  # vertical nw|sw
                # assert edges_bdy0[ (n-1)*m + n*i + j ] == nw_idx
                # assert edges_bdy1[ (n-1)*m + n*i + j ] == sw_idx
                faces_bdy2.append(edges_rep[-1])  # most recent edge is ne\sw
                # assert edges_bdy0[ edges_rep[-1] ] == ne_idx
                # assert edges_bdy1[ edges_rep[-1] ] == sw_idx
    
                # Make face (sw,se,ne).
                faces_hgt.append(np.max([sw_hgt, se_hgt, ne_hgt]))
                faces_rep.append(len(faces_rep))
                faces_bdy0.append((n-1)*(i+1) + j)  # horizontal sw-se
                # assert edges_bdy0[ (n-1)*(i+1) + j ] == sw_idx
                # assert edges_bdy1[ (n-1)*(i+1) + j ] == se_idx
                faces_bdy1.append((n-1)*m + n*i + j+1)  # vertical ne|se
                # assert edges_bdy0[ (n-1)*m + n*i + j+1 ] == ne_idx
                # assert edges_bdy1[ (n-1)*m + n*i + j+1 ] == se_idx
                faces_bdy2.append(edges_rep[-1])  # most recent edge is ne\sw
                # assert edges_bdy0[ edges_rep[-1] ] == ne_idx
                # assert edges_bdy1[ edges_rep[-1] ] == sw_idx
    
    verts_pos = np.ones_like(verts_hgt, dtype='bool')
    edges_pos = np.ones_like(edges_rep, dtype='bool')

    verts = pd.DataFrame({'height': verts_hgt,
                          'rep': verts_rep,
                          'pos': verts_pos},
                         columns=['height', 'pos', 'rep'])

    edges = pd.DataFrame({'height': edges_hgt,
                          'rep': edges_rep,
                          'pos': edges_pos,
                          'bdy0': edges_bdy0,
                          'bdy1': edges_bdy1},
                         columns=['height', 'pos', 'rep', 'bdy0', 'bdy1'])

    stratum = {0: verts, 1:edges}
    if diagonals_and_faces:
        faces_pos = np.ones_like(faces_rep, dtype='bool')
        faces = pd.DataFrame({'height': faces_hgt,
                              'rep': faces_rep,
                              'pos': faces_pos,
                              'bdy0': faces_bdy0,
                              'bdy1': faces_bdy1,
                              'bdy2': faces_bdy2},
                             columns=['height', 'pos', 'rep', 'bdy0', 'bdy1', 'bdy2'])
        stratum[2] = faces

    return SimplicialComplex(stratum=stratum)


class SimplicialComplex(object):
    r"""
    A class for abstract *weighted* simplicial complexes.
    A SimplicialComplex is built from 0-cells (vertices), 1-cells (edges),
    2-cells (faces), and so on.

    Each cell knows its boundary.  A 0-cell has no boundary. A 1-cell has two
    0-cells as its boundary.  A 2-cell has three 1-cells as its boundary, and
    so on.

    Each cell *must* height value, called `height`.  These heights are used in
    several topological algorithms that depend on filtering.

    Each cell *may* have a mass value, called `mass`.  These masses are used in
    some data-analysis methods that involve weighted averaging or probability.

    Each cell can
    A SimplicialComplex has no notion of coordinates or embedding.  For that,
    use the :class:`PointCloud` class, which inherits all methods from
    SimplicialComplex but also adds coordinate-dependent methods.

    Parameters
    ----------
    stratum : dict
        Dictionary of :class:`pandas.DataFrame` objects holding vertices,
        edges, faces, and so on.  See examples below.

    Notes
    -----
    One can reference the :class:`Simplex` objects that are separated by
    dimension into :class:`SimplexStratum` objects.

    Each :class:`Simplex` object has a height, so these are *filtered*
    simplicial complexes.

    Each :class:`Simplex` object may have a mass, so these can be are *weighted*
    simplicial complexes.

    The **rep** and **pos** attributes are used when computing various
    homologies, such as the Rips or Cech complexes.

    Whenever possible, computations are done on :class:`numpy.ndarray` arrays
    in compiled code, so they are usually quite fast.

    Examples
    --------
    For example, a formal triangle could be built this way:

    >>> vertices = pd.DataFrame({'height':[ 0.0, 0.0, 0.0],
    ...                          'mass': [1.0, 1.0, 1.0],
    ...                          'pos': [True, True, True],
    ...                          'rep' : [0, 1, 2]})
    >>> edges = pd.DataFrame({'height':[ 0.0, 0.0, 0.0],
    ...                       'pos': [True, True, True],
    ...                       'rep' : [0, 1, 2],
    ...                       'bdy0': [0, 1, 2],
    ...                       'bdy1': [1, 2, 0]})
    >>> faces = pd.DataFrame({'height': [0.0],
    ...                       'pos': [True],
    ...                       'rep': [0],
    ...                       'bdy0': [0],
    ...                       'bdy1': [1],
    ...                       'bdy2': [2]})
    >>> T = SimplicialComplex(stratum = {0: vertices, 1: edges, 2: faces})
    >>> print(T)
    A SimplicialComplex with 3 points, 3 edges, and 1 faces.
    """

    def __init__(self, stratum=None):
        if stratum is not None:
            assert all(type(k) == int and k >= 0 for k in stratum.keys()), \
                "The strata of a SimplicialComplex must be indexed by non-negative integers."
            self.stratum = stratum
        else:
            self.stratum = {0: stratum_maker(0), 1: stratum_maker(1)}

        self._nn = dict()
        self._cellstratum = dict()
        self.pers0 = None
        self.pers1 = None

    @classmethod
    def from_distances(cls, dists, max_length=-1.0, points=None):
        r"""
        Construct a `SimplicialComplex` from a symmetric matrix of distances.

        Parameters
        ----------
        dists : `numpy.ndarray`
            An N-by-N symmetric array, with 0s on the diagonal,
            as obtained from :func:`scipy.spatial.distances.squareform`

        max_length : float
            If :code:`max_length >= 0`, store only those edges of length less
            than :code:`max_length`.  Default: -1, store all edges.

        points : `pandas.DataFrame`
            A fully-formed DataFrame of point information for
            stratum[0]. But, if you have that info, you probably want to use
            `PointCloud` instead.

        Returns
        -------
        `SimplicialComplex`
        """
        stratum = stratum_from_distances(dists, max_length, points)
        return cls(stratum=stratum)

    def check(self):
        r"""Run consistency checks on all simplices in all dimensions.
        raises ValueError if anything is wrong.
        """
        for dim in self.stratum.keys():
            if dim > 0:
                valcheck = fast_algorithms.check_heights(self, dim)
                if not valcheck == -1:
                    raise ValueError("Not a filtration! Check 'height' in ({}).stratum[{}].iloc[{}]".format(self, dim, valcheck))

    def __repr__(self):
        f = 0
        if 2 in self.stratum:
            f = len(self.stratum[2])
        return f"A SimplicialComplex with {len(self.stratum[0])} points, {len(self.stratum[1])} edges, and {f} faces."

    def cells(self, dim):
        r""" iterate over all :class:`Simplex` objects of dimension dim.
        This is generated on-demand from the :class:`SimplexStratum`.
        """
        if dim not in self._cellstratum:
            self._cellstratum[dim] = SimplexStratum(self, dim)
        return self._cellstratum[dim]

    def reset(self):
        """ delete persistence diagrams, and forget all representative and
        positivity information.  Use this before re-running :func:`make_pers0`
        or other methods in `homology` with a smaller cutoff.
        """
        self.pers0 = None
        self.pers1 = None
        for dim in self.stratum.keys():
            if self.stratum[dim] is None:
                self.stratum[dim] = stratum_maker(dim)
            self.stratum[dim]['rep'].values[:] = self.stratum[
                dim].index.values  # identity representation
            self.stratum[dim]['pos'].values[:] = True
        pass

    def make_pers0(self, cutoff=-1.0, show_diagonal=False, until_connected=(0,0)):
        r"""Run the UnionFind algorithm to mark connected components of the
        SimplicialComplex.  This marks points as positive/negative.
        It also marks the reprensetatives of points.
        It makes a PersDiag object with unionfind, saved as :code:`self.pers0`
        """

        if (1 not in self.stratum.keys()) or len(self.stratum[1]) == 0:
            raise ValueError("This SimplicialComplex has no 1-stratum (edges).  Persistence is meaningless.")

        try:
            if self.max_length >= 0.0 and cutoff > self.max_length:
                raise ValueError("Persistence cutoff is greater than max_length of pre-computed edges.  This is meaningless.")
        except AttributeError:
            pass

        
        if not np.all(np.diff(self.stratum[1]['height'].values) >= 0):
            print("Edges must be sorted by length!  Re-sorting.")
            self.stratum[1].sort_values(by=['height', 'bdy0', 'bdy1'],
                                        ascending=[True, True, True],
                                        inplace=True)
            print("sorted") 

        tbirth_index, tdeath_index, ybirth_index, ydeath_index, mergetree = homology.dim0.unionfind(self, 
            np.float64(cutoff), np.int64(show_diagonal),
            np.int64(until_connected[0]), np.int64(until_connected[1]))
        self.pers0 = homology.PersDiag(tbirth_index, tdeath_index, ybirth_index, ydeath_index, mergetree)
        pass

    def sever(self):
        r"""
        Subdivide a SimplicialComplex or PointCloud into several smaller
        partitions, using the known 0-dimensional persistence diagram.  This is
        an iterator (it _yields_ the terms).

        Two points end up in the same partition if and only if they are
        connected by a sequence of edges of length < cutoff.

        Yields
        ------
        pairs (indices, subpointcloud) of persistently connected
        SimplicialComplexes/PointClouds.
        The first term gives indices of the these points from the original `PointCloud`
        The second term gives a new `PointCloud` with its own sequential index.

        Notes
        -----
        This uses the 0-dimensional Persistence Diagram; therefore, you should
        run `self.reset()` and `self.make_pers0(cutoff)` first.

        See Also
        --------
        :func:`make_pers0` :func:`reset`


        Examples
        --------

        >>> pc = PointCloud(np.array([[0.,0.],[0.,0.5],[1.,0.],[5.,0.],[6.,0.],[5.,-0.6]]), max_length=-1.0)
        >>> pc.make_pers0(cutoff=1.9)
        >>> pc.stratum[0]
           height  mass    pos  rep
        0     0.0   1.0   True    0
        1     0.0   1.0  False    0
        2     0.0   1.0  False    0
        3     0.0   1.0   True    3
        4     0.0   1.0  False    3
        5     0.0   1.0  False    3
        >>> pc.stratum[1]
              height    pos  rep  bdy0  bdy1
        0   0.500000  False    0     0     1
        1   0.600000  False    1     3     5
        2   1.000000  False    2     0     2
        3   1.000000  False    3     3     4
        4   1.118034   True    4     1     2
        5   1.166190   True    5     4     5
        6   4.000000   True    6     2     3
        7   4.044750   True    7     2     5
        8   5.000000   True    8     0     3
        9   5.000000   True    9     2     4
        10  5.024938   True   10     1     3
        11  5.035871   True   11     0     5
        12  5.119570   True   12     1     5
        13  6.000000   True   13     0     4
        14  6.020797   True   14     1     4
        >>> for indices,sub_pc in pc.sever():
        ...     print(indices)
        ...     print(sub_pc)
        ...     print(sub_pc.stratum[0])
        ...     print(sub_pc.stratum[1])
        [0 1 2]
        A SimplicialComplex with 3 points, 3 edges, and 0 faces.
           height  mass    pos  rep
        0     0.0   1.0   True    0
        1     0.0   1.0  False    0
        2     0.0   1.0  False    0
             height    pos  rep  bdy0  bdy1
        0  0.500000  False    0     0     1
        2  1.000000  False    2     0     2
        4  1.118034   True    4     1     2
        [3 4 5]
        A SimplicialComplex with 3 points, 3 edges, and 0 faces.
           height  mass    pos  rep
        3     0.0   1.0   True    3
        4     0.0   1.0  False    3
        5     0.0   1.0  False    3
            height    pos  rep  bdy0  bdy1
        1  0.60000  False    1     3     5
        3  1.00000  False    3     3     4
        5  1.16619   True    5     4     5

        """

        from homology.dim0 import all_roots

        assert np.all(self.stratum[0].index.values ==
        np.arange(len(self.stratum[0].index))), "Indexing was inconsistent. Watch out for re-indexing within sub-clusters of sever!"
        roots = self.stratum[0]['rep'].values.copy()
        all_roots(roots)
        # we rebuild the strata by filtering the vertices and then propagating
        # the filter up dimension-by-dimension
        for this_root in np.where(self.stratum[0]['pos'].values == True)[0]:
            new_strata = dict()
            
            this_condition = (roots == this_root)
            this_point_indices = np.where(this_condition)[0]
            new_strata[0] = self.stratum[0].loc[this_point_indices]
            # CHECK INDEX HERE between bool and index 
            for d, stratum_d in self.stratum.items():
                    if d > 0:
                        boundaries = stratum_d[[f'bdy{i}' for i in range(d+1)]].values
                        matches = np.ndarray(boundaries.shape, dtype='bool')
                        matches[:,:] = this_condition[boundaries[:,:]] 
                        this_condition = np.all(matches, axis=1) # this is now indexed against this dimension's simplices!
                        this_simplex_indices = np.where(this_condition)[0]
                        new_strata[d] = self.stratum[d].loc[this_simplex_indices]
            yield this_point_indices, SimplicialComplex(new_strata)


    def make_pers1_rca1(self, cutoff=-1.0):
        r""" Run RCA1 and make a 1-dimensional `homology.PersDiag` for the
        edge-pairings for cycle generators.

        This reruns self.make_pers0(cutoff) again, to make sure components are
        marked correctly.

        Parameters
        -----------
        cutoff: float
            Maximum edge height to use for RCA1 algorithm. Higher edges ignored.
            (Default: -1, meaning use all edges.)

        Returns
        -------
        none.  Produces `self.pers1`

        Table of edge pairs, similar to a persistence diagram.

        BUGS
        ----
        data = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,0.5]]) fails.



        Examples
        --------

        >>> data = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
        >>> pc = PointCloud(data, max_length=-1)
        >>> print(pc.stratum[1])
             height   pos  rep  bdy0  bdy1
        0  1.000000  True    0     0     1
        1  1.000000  True    1     0     2
        2  1.000000  True    2     1     3
        3  1.000000  True    3     2     3
        4  1.414214  True    4     0     3
        5  1.414214  True    5     1     2
        >>> pc.make_pers1_rca1()
        >>> print(pc.pers1.diagram)
           birth_index  death_index  birth     death      pers
        0            3            4    1.0  1.414214  0.414214

        >>> data = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,0.5]])
        >>> pc = PointCloud(data, max_length=-1)
        >>> print(pc.stratum[1])
             height   pos  rep  bdy0  bdy1
        0  0.500000  True    0     1     3
        1  1.000000  True    1     0     1
        2  1.000000  True    2     0     2
        3  1.118034  True    3     0     3
        4  1.118034  True    4     2     3
        5  1.414214  True    5     1     2
        >>> pc.make_pers1_rca1()
        >>> print(pc.pers1.diagram)
        Empty DataFrame
        Columns: [birth_index, death_index, birth, death, pers]
        Index: []

        """
        # we need 0dim persistence first.
        self.reset()
        self.make_pers0(cutoff=cutoff)

        column_list, column_edge_index, stop_edge = homology.dim1.rca1(self.stratum[1], cutoff=cutoff)

        assert len(column_list) == len(column_edge_index)

        pers_list = [(c[-1], column_edge_index[i]) for i, c in
                     enumerate(column_list) if c]
        p = np.array(pers_list)
        if len(p)>0:
            mergetree = dict([]) # we can't compute a mergetree yet
            self.pers1 = homology.PersDiag(
                p[:, 0],
                p[:, 1],
                self.stratum[1]['height'].loc[p[:, 0]].values,
                self.stratum[1]['height'].loc[p[:, 1]].values,
                mergetree)

        else:
            # no births or deaths recorded
            self.pers1 = homology.PersDiag([], [], [], [], dict([]))

        pass


class PointCloud(SimplicialComplex):
    r""" PointCloud is a class for *embedded*, weighted simplicial complexes.
    This is a subclass of :class:`SimplicialComplex`, with the additional property
    that every 0-cell (vertex) is actually a point in :math:`\mathbb{R}^k.`

    The most basic and most common example of a PointCloud is an indexed set of
    :math:`N` points in :math:`\mathbb{R}^k` with heights assigned as 0, and
    mass assigned as 1.

    Typically, a user starts with 0-cells only.  Then, any 1-cells, 2-cells,
    ..., are created later via some topological construction.
    """

    def __init__(self, data_array, max_length=0.0, heights=None, masses=None,
                 dist='euclidean', idx0=None, cache_type=None):
        r""" Construct a :class:`PointCloud` from a cloud of n points in
        :math:`\mathbb{R}^k.`

        Parameters
        ----------
        data_array : :class:`numpy.ndarray`
            A np array with shape=(n,k), to use as the pointcloud.  The array
            must have dtype=='float64'.

        max_length : float
            If max_length is positive, then find edges of length <= max_length.
            This uses the :class:`multidim.covertree.CoverTree` for efficiency.
            Default is 0.0, meaning compute no edges.

        heights : :class:`numpy.ndarray`
            If heights is given, it is used to assign graded values to the
            points in data_array. It must be a np array of dtype=='float64' and
            shape==(n,), like from np.apply_along_axis(my_func, 1, data_array)
            Default: None (all have height 0.0)

        masses : :class:`numpy.ndarray`
            If masses is given, it is used to assign mass values to the
            points in data_array. It must be a np array of dtype=='float64' and
            shape==(n,), like from np.apply_along_axis(my_func, 1, data_array)
            Default: None (all have mass 1.0)

        idx0 : :class:`numpy.ndarray`
            If idx0 is given, it is used to assign index values to the
            points in data_array. It must be a np array of dtype=='int64' and
            shape==(n,),
            Default: None (index by order given in data_array)

        cache_type : None or "np" or "dict"
            What type of distance cache to use.  Often None is actually faster!
            If you really care about speed, remember to use -O

        dist : function
            If dist is given, it is used as the distance function for computing
            edge lengths, via scipy.spatial.distance.pdist.  Not used with on-demand caching.
            Default: 'euclidean'


        """
        assert data_array.dtype == np.float64, "Data must be float64."
        n, k = data_array.shape
        self.dimension = k

        self.cache_type = cache_type
        self.dist_cache = None
        self.dist = dist

        if self.cache_type is None:
            self.dist_cache = None
        elif self.cache_type == "np":
            self.dist_cache = np.eye(n, dtype=np.float64) - 1.0
        elif self.cache_type == "dict":
            self.dist_cache = dict(((i,i), np.float64(0.0)) for i in range(n))
        else:
            raise ValueError("cache_type can be None or 'dict' or 'np'")

        if heights is None:
            heights = np.zeros(n, dtype=np.float64)
        else:
            assert type(heights) == np.ndarray \
                   and heights.shape == (n,) \
                   and heights.dtype == 'float64', \
                   "Wrong type or size for heights data on pointcloud."

        hgt0 = heights

        if masses is None:
            masses = np.ones(n, dtype=np.float64)
        else:
            assert type(masses) == np.ndarray \
                   and masses.shape == (n,) \
                   and masses.dtype == 'float64', \
                   "Wrong type or size for heights data on pointcloud."

        mas0 = masses

        pos0 = np.ones(shape=(n,), dtype='bool')
        if idx0 is None:
            idx0 = np.arange(n, dtype='int64')
        else:
            assert type(idx0) == np.ndarray \
                   and idx0.shape == (n,) \
                   and idx0.dtype == 'int64', \
                   "Wrong type or size for indexing data on pointcloud."

        points = pd.DataFrame({
            'height': hgt0,
            'mass': mas0,
            'pos': pos0,
            'rep': idx0,
        },
                columns=['height', 'mass', 'pos', 'rep'],
                index=idx0)

        self.coords = pd.DataFrame(data_array, index=idx0)
        self.covertree = None

        edges = stratum_maker(1)
        super(self.__class__, self).__init__(stratum={0: points, 1: edges})

        self.labels = np.zeros(shape=(self.coords.shape[0],), dtype=np.int64)
        self.source = np.zeros(shape=(self.coords.shape[0],), dtype=np.int64)
        self.label_info = pd.DataFrame(index=['black'])
        self.label_info['clouds'] = np.array([1], dtype=np.int64)
        self.label_info['points'] = np.array([n], dtype=np.int64)
        self.label_info['tot_mass'] = np.array([self.stratum[0]['mass'].sum()])
        self.label_info['int_index'] = np.array([0], dtype=np.int64)

        self.max_length = max_length
        if self.max_length > 0.0 or self.max_length == -1.0:
            # use covertree to make all appropriate edges.
            from . import covertree
            self.covertree = covertree.CoverTree(self)
            bdy0 = []
            bdy1 = []
            hgts = []
            for i, j, d in self.covertree.make_edges(max_distance=self.max_length):
                bdy0.append(min(i,j))
                bdy1.append(max(i,j))
                hgts.append(d)
            bdy0 = np.array(bdy0, dtype=np.int64)
            bdy1 = np.array(bdy1, dtype=np.int64)
            hgts = np.array(hgts)
            sortby = hgts.argsort()
            bdy0 = bdy0[sortby]
            bdy1 = bdy1[sortby]
            hgts = hgts[sortby]

            edges = pd.DataFrame({'height': hgts,
                                  'pos': np.ones(shape=hgts.shape, dtype='bool'),
                                  'rep': np.arange(hgts.shape[0], dtype=np.int64),
                                  'bdy0': bdy0, 'bdy1': bdy1, },
                                 columns=['height', 'pos', 'rep', 'bdy0', 'bdy1'],
                                 index=np.arange(hgts.shape[0], dtype=np.int64))
            self.stratum[1] = edges

    @classmethod
    def from_distances(cls, *args, **kwargs):
        r"""
        This method is not available for `PointCloud`, because actual
        coordinates are needed.   Perhaps you want to use
        :func:`SimplicialComplex.from_distances` instead?
        """
        raise NotImplementedError("This method does not inherit to PointCloud.  Use the version from the parent class, SimplicialComplex.")


    def sever(self):
        r""" This calls the sever() command of the parent class,
        :class:SimplicialComplex, but also reproduces the PointCloud as
        appropriate. """
        for idx, cx in super(PointCloud, self).sever():
            subcoords = self.coords.loc[idx].values
            subpc = PointCloud(subcoords, idx0 = idx)
            subpc.stratum = cx.stratum
            subpc.labels = self.labels[idx]  # warning!  indexing my get off!
            yield idx, subpc

    def plot(self, canvas, cutoff=-1, color='purple', pos_edges=False,
             edge_alpha=-1.0, size=1,
             twocells=False, title="SimplicialComplex", label=False):
        r"""
        Plot a PointCloud, decorated by various proeprties.

        Often slow!

        Parameters
        ----------
        canvas : object
            An instance of
            `bokeh.plotting.figure.Figure` as in
            :code:`canvas = bokeh.plotting.figure()`
            or an instance of :class:`matplotlib.axes._subplots.AxesSubplot` as
            in :code:`axes,canvas = matplotlib.pyplot.subplots()`

        cutoff : float
            if cutoff>=0, only draw edges up to length <cutoff

        twocells : boolean
            draw 2-cells (triangles)?

        title : string
            title for plot

        label : boolean
            label points in plot?

        """

        if type(canvas).__module__ == 'bokeh.plotting.figure':
            from bokeh.models import Range1d
            canvas_type = "bokeh"
            import bokeh.plotting
        elif type(canvas).__module__ == 'matplotlib.axes._subplots':
            canvas_type = "pyplot"
            import matplotlib.pyplot as plt
        else:
            raise NotImplementedError(
                    "canvas must be a bokeh.plotting.figure() or a matplotlib.pyplot.subplots()[1].  You gave me {}".format(
                        type(canvas))
            )

        n, k = self.coords.shape
        assert k == 2, "I can only plot in R^2. Maybe project your data first?"

        if canvas_type == "bokeh":
            canvas.title.text = title
        elif canvas_type == "pyplot":
            canvas.set_title(title)

        if twocells:
            raise NotImplementedError(
                "Have not re-incomporated 2-cells into RCA1 yet.")

        # find edges
        all_edges = self.stratum[1]
        if cutoff >= 0:
            all_edges = all_edges[all_edges['height'] < cutoff]

        if len(all_edges) > 0:
            minhgt = np.min(all_edges['height'].values)
            maxhgt = np.max(all_edges['height'].values)
        else:
            edge_alpha = 1.0

        # plot positive edges, need to build structure for multi_line
        if pos_edges:
            pos = all_edges[all_edges['pos'] == True]
            val = pos['height'].values
            pt0 = self.coords.loc[pos['bdy0'].values].values
            pt1 = self.coords.loc[pos['bdy1'].values].values
            pts = np.hstack([pt0, pt1])
            xs = pts[:, 0::2]
            ys = pts[:, 1::2]

            if canvas_type == "bokeh":
                canvas.multi_line(list(xs),
                                  list(ys),
                                  line_width=1, alpha=0.4, color='orange')
            elif canvas_type == "pyplot":
                for i in range(xs.shape[0]):
                    if edge_alpha >= 0.0:
                        this_edge_alpha = edge_alpha
                    else:
                        this_edge_alpha = 0.5 + 0.5*(val[i] - minhgt)/(maxhgt - minhgt)
                    # should use Collections instead.
                    canvas.plot(xs[i, :], ys[i, :],
                        alpha=this_edge_alpha, color='orange')

        # plot negative edges, need to build structure for multi_line
        neg = all_edges[all_edges['pos'] == False]
        val = neg['height'].values

        pt0 = self.coords.loc[neg['bdy0'].values].values
        pt1 = self.coords.loc[neg['bdy1'].values].values
        pts = np.hstack([pt0, pt1])
        xs = pts[:, 0::2]
        ys = pts[:, 1::2]

        if canvas_type == "bokeh":
            canvas.multi_line(list(xs),
                              list(ys),
                              line_width=1, alpha=0.6, color='blue')
        elif canvas_type == "pyplot":
            for i in range(xs.shape[0]):
                # should use Collections instead.
                if edge_alpha >= 0.0:
                    this_edge_alpha = edge_alpha
                else:
                    this_edge_alpha = 0.5 + 0.5*(val[i] - minhgt)/(maxhgt - minhgt)
                    # should use Collections instead.
                canvas.plot(xs[i, :], ys[i, :],
                    alpha=this_edge_alpha, color='blue')

        all_verts = self.stratum[0]
        # CURRENT UNIONFIND DOES NOT MARK NEG VERTS
        neg = all_verts[all_verts['pos'] == False]
        xs = list(self.coords.loc[neg.index, 0])
        ys = list(self.coords.loc[neg.index, 1])
        if canvas_type == "bokeh":
            canvas.circle(xs, ys, color='black', alpha=0.5, size=size)
        elif canvas_type == "pyplot":
            canvas.scatter(x=xs, y=ys, s=size, color='black', alpha=0.5)

        pos = all_verts[all_verts['pos'] == True]
        xs = self.coords.loc[pos.index, 0]
        ys = self.coords.loc[pos.index, 1]
        cs = list(self.label_info.index[self.labels[np.where(all_verts['pos'] == True)[0]]])
        if canvas_type == "bokeh":
            # fix the aspect ratio!
            xmid = (xs.max() + xs.min())/2.0
            ymid = (ys.max() + ys.min())/2.0
            span = max([xs.max() - xmid,
                        xmid - xs.min(),
                        ys.max() - ymid,
                        ymid - ys.min()])
            canvas.x_range = Range1d(xmid-span, xmid+span)
            canvas.y_range = Range1d(ymid-span, ymid+span)
            canvas.circle(list(xs), list(ys), color=cs, alpha=0.4, size=size)

        elif canvas_type == "pyplot":
            canvas.scatter(x=xs, y=ys, color=cs, alpha=0.4, s=size)

        if label:
            if canvas_type == "bokeh":
                canvas.text(xs, ys, list(map(str, list(pos.index.values))))

        pass

    def gaussian_fit(self, center=None):
        r"""
        Fit a normalized Gaussian to this cloud (using SVD).


        Parameters
        ----------
        center
            If center is None (default), we find the best Gaussian with free mean.
            If center is given as an integer, use the point with that integer as the
            mean of the Gaussian.
            If center is given as a tuple or array, use that coordinate point as
            the mean of the Gaussian.

        Returns
        -------
            (mean, sigma, rotation) for the Gaussian, suitable for `gaussian`

        """
        return fast_algorithms.gaussian_fit(self.coords.values, center)

    def cache_usage(self):
        r""" Compute the size of the distance cache.  That is, what
        fraction of distances have we computed so far? """
        n = self.coords.values.shape[0]
        n_choose_2 = 0.5*n*(n-1)

        if self.cache_type is None:
            return 0.0
        elif self.cache_type == "np":
            computed = np.count_nonzero(self.dist_cache >= 0)
            return (computed - n)/n_choose_2
        elif self.cache_type == "dict":
            computed = len(self.dist_cache)
            return (computed - n)/n_choose_2

    def nearest_neighbors_slow(self, k):
        r""" Compute k nearest-neighbors of the PointCloud, by brute-force.
        Answers are cached in `self._nn[k]`

        Parameters
        ----------
        k: int
            How many nearest neighbors to compute

        Returns
        -------
        np array with dtype int and shape==(N,k+1).  Entry [i,j] is the jth
        nearest neighbor of vertex i.  Note that entry [i,0] == i, so [i,k]
        is the kth nearest neighbor.

        Notes
        -----
        This method is intended for testing, and should only be used on small datasets.
        On a random example with 1,000 points in :math:`\mathbb{R}^2:` seeking `k=5` nearest
        neighbors, this method takes at least twice as long as :func:`nearest_neighbors`, and the
        discrepency is roughly quadratic.  On 2,000 points, it is about 4 times slower.

        Examples
        --------

        >>> pc = PointCloud(np.array([[ 0.58814682,  0.45405299],
        ...                           [ 0.09197879,  0.39721367],
        ...                           [ 0.29128654,  0.28372039],
        ...                           [ 0.14593167,  0.7027367 ],
        ...                           [ 0.77068438,  0.37849037],
        ...                           [ 0.17281855,  0.70204687],
        ...                           [ 0.48146217,  0.54619034],
        ...                           [ 0.27831744,  0.67327757],
        ...                           [ 0.49074255,  0.70847318],
        ...                           [ 0.132656,    0.0860524 ]]))
        >>> pc.nearest_neighbors_slow(3)
        array([[0, 6, 4, 8],
               [1, 2, 3, 9],
               [2, 1, 9, 6],
               [3, 5, 7, 1],
               [4, 0, 6, 8],
               [5, 3, 7, 1],
               [6, 0, 8, 7],
               [7, 5, 3, 8],
               [8, 6, 7, 0],
               [9, 2, 1, 6]])

        See Also
        --------
        :func:`multidim.PointCloud.nearest_neighbors`
        """

        # simple cache
        if k in self._nn:
            return self._nn[k]

        num_points = self.coords.shape[0]
        self._nn[k] = np.ndarray(shape=(num_points, k+1), dtype=np.int64)

        all_points = self.coords.index.values
        dists = self.dists(all_points, all_points)
        self._nn[k] = dists.argsort(axis=1)[:, :k+1]  # 0th entry is always self.
        return self._nn[k]

    def nearest_neighbors(self, k):
        r""" Compute k nearest-neighbors of the PointCloud, using a clever CoverTree algorithm.
        Answers are cached in `self._nn[k]`

        Parameters
        ----------
        k: int
            How many nearest neighbors to compute

        Returns
        -------
        np array with dtype int and shape==(N,k+1).  Entry [i,j] is the jth
        nearest neighbor of vertex i.  Note that entry [i,0] == i, so [i,k]
        is the kth nearest neighbor.

        Examples
        --------

        >>> pc = PointCloud(np.array([[ 0.58814682,  0.45405299],
        ...                           [ 0.09197879,  0.39721367],
        ...                           [ 0.29128654,  0.28372039],
        ...                           [ 0.14593167,  0.7027367 ],
        ...                           [ 0.77068438,  0.37849037],
        ...                           [ 0.17281855,  0.70204687],
        ...                           [ 0.48146217,  0.54619034],
        ...                           [ 0.27831744,  0.67327757],
        ...                           [ 0.49074255,  0.70847318],
        ...                           [ 0.132656,    0.0860524 ]]))
        >>> pc.nearest_neighbors(3)
        array([[0, 6, 4, 8],
               [1, 2, 3, 9],
               [2, 1, 9, 6],
               [3, 5, 7, 1],
               [4, 0, 6, 8],
               [5, 3, 7, 1],
               [6, 0, 8, 7],
               [7, 5, 3, 8],
               [8, 6, 7, 0],
               [9, 2, 1, 6]])


        See Also
        --------
        :func:`multidim.PointCloud.nearest_neighbors_slow`

        """
        from . import covertree

        if self.covertree is None:
            self.covertree = covertree.CoverTree(self)

        # simple cache
        if k in self._nn:
            return self._nn[k]

        num_points = self.coords.shape[0]

        # -1 means "not found yet"
        self._nn[k] = -np.ones(shape=(num_points, k+1), dtype=np.int64)

        # Make sure we have the entire covertree.
        levels = [ level for level in self.covertree ]

        # run backwards:
        for level in reversed(levels):
            r = level.radius
            for ci in level.adults:
                for x in level.children[ci]:
                    unknown_neighbors = np.where(self._nn[k][x] < 0)[0]
                    if len(unknown_neighbors) > 0:
                        to_find = unknown_neighbors[0]

                        candidates = []
                        for cj in level.friends1[ci]:
                            candidates.extend(level.children[cj])
                        candidates = np.array(candidates)

                        num_found = min(k+1, len(candidates))

                        # don't bother computing lengths if there is nothing to
                        # learn
                        if num_found >= to_find:

                            dists = fast_algorithms.distance_cache_None(
                                        np.array([x]), candidates,
                                        self.coords.values).flatten()
                            order = dists.argsort()
                            self._nn[k][x, to_find:num_found] = candidates[order][to_find:num_found]

        return self._nn[k]

#
#        all_points = self.coords.index.values
#        dists = self.dists(all_points, all_points)
#        self._nn[k] = dists.argsort(axis=1)[:, :k+1]  # 0th entry is always self.
#        return self._nn[k]

    def witnessed_barycenters(self, k):
        r""" Build the PointCloud of k-witnessed barycenters, weighted by
        distance-to-measure. This calls :func:`nearest_neighbors` with argument
        :code:`(k-1)`, which can be slow.

        Parameters
        ----------
        k : int
            How many vertices for each witnessed barycenter.  That is, use
            the (k-1) nearest neighbors, along with the vertex itself.

        Returns
        -------
        pc : :class:`PointCloud`
            A pointcloud whose 0-cells are the witnessed barycenters, and
            whose 1-cells are the edges between
            those barycenters, all weighted by the notion of distance to a
            measure.
        """

        n, d = self.coords.values.shape

        # First, look at the indices to uniqify
        polygons_indices = [tuple(np.sort(verts)) for verts in self.nearest_neighbors(k-1)]
        polygons_indices = np.array(list(set(polygons_indices)))

        p = polygons_indices.shape[0]
        assert p <= n

        # build the polygons from coordinates
        polygons = []
        for points in polygons_indices:
            polygons.append(self.coords.values[points, :])
        polygons = np.array(polygons)

        # Find the vertex barycenters
        assert polygons.shape == (p, k, d)
        polygons = polygons.swapaxes(0, 1)
        assert polygons.shape == (k, p, d)
        barycenters = polygons.mean(axis=0)
        assert barycenters.shape == (p, d)

        # compute weights
        diffs = polygons - barycenters
        assert diffs.shape == (k, p, d)
        norms = np.linalg.norm(diffs, axis=2)**2
        assert norms.shape == (k, p)
        weights = -norms.sum(axis=0)/k
        assert weights.shape == (p,)
        pcbc = PointCloud(barycenters,
                          heights=np.sqrt(-weights),
                          dist=self.dist)
        pcbc.dists = squareform(pdist(pcbc.coords.values, pcbc.dist))

        # make edges
        hgt = []
        pos = []
        idx = []
        bdy0 = []
        bdy1 = []
        for e, (i, j) in enumerate(itertools.combinations(pcbc.stratum[0].index.values, 2)):
            idx.append(e)
            bdy0.append(i)
            bdy1.append(j)
            pos.append(True)
            mu = pcbc.dists[i, j]
            wi = weights[i]
            wj = weights[j]
            r = np.sqrt(mu**2*(mu**2 - 2*wi - 2*wj) + (wi - wj)**2)/2/mu
            hgt.append(r)

        edges = pd.DataFrame({
            'height': hgt,
            'pos': pos,
            'rep': idx,
            'bdy0': bdy0,
            'bdy1': bdy1,
            },
                columns=['height', 'pos', 'rep', 'bdy0', 'bdy1'],
                index=idx)
        pcbc.stratum[1] = edges
        return pcbc

    def unique_with_multiplicity(self):
        r"""
        Look for duplicate points, and mark their multiplicity.
        This sets self.multiplicity


        ToDo:  Use Covertree.

        Examples
        --------

        >>> a = np.array([[5.0, 2.0], [3.0, 4.0], [5.0, 2.0]])
        >>> pc = PointCloud(a)
        >>> b, counts = pc.unique_with_multiplicity()
        >>> print(b)
        [[ 3.  4.]
         [ 5.  2.]]
        >>> print(counts)
        [1 2]
        """

        coords = self.coords.values
        assert coords.shape[1] == 2,\
            "This uniqifying method can use only 2-dim data"

        s = coords.shape
        assert coords.dtype == 'float64'
        coords.dtype = 'complex128'
        tmp_coords, tmp_index, tmp_inverse, tmp_counts = np.unique(
            coords.flatten(),
            return_index=True,
            return_inverse=True,
            return_counts=True)
        tmp_coords.dtype = 'float64'
        coords.dtype = 'float64'
        assert coords.shape == s
        n = tmp_coords.shape[0]
        d = coords.shape[1]
        assert n % d == 0
        tmp_coords.shape = (n//d, d)

        return tmp_coords, tmp_counts

    def dists(self, indices0, indices1):
        r""" Compute distances points indices0 and indices1.
        indices0 and indices1 must be 1-dimensional np arrays,
        so use not "5" but "np.array([5])"

        The return is a np array with shape == (len(indices0), len(indices1))

        This uses the distance cache, depending on self.cache_type.
        You can query the size of the cache with :func:`cache_usage`.

        """
        # Allow boolean selectors
        if indices0.dtype == np.uint8:
            indices0 = np.where(indices0)[0]
        if indices1.dtype == np.uint8:
            indices1 = np.where(indices1)[0]

        if self.cache_type is None:
            return fast_algorithms.distance_cache_None(indices0,
                                                       indices1,
                                                       self.coords.values)
        elif self.cache_type == "np":
            N = self.coords.values.shape[0]
            if self.dist_cache is None or self.dist_cache.shape != (N, N):
                self.dist_cache = np.eye(N, dtype=np.float64) - 1.0
            return fast_algorithms.distance_cache_numpy(indices0,
                                                     indices1,
                                                     self.coords.values,
                                                     self.dist_cache)
        elif self.cache_type == "dict":
            N = self.coords.values.shape[0]
            if self.dist_cache is None:
                self.dist_cache = dict(((i, i), np.float64(0.0)) for i in range(N))
            return fast_algorithms.distance_cache_dict(indices0,
                                                       indices1,
                                                       self.coords.values,
                                                       self.dist_cache)
        else:
            raise ValueError("cache_type can be None or 'dict' or 'np'")

    def cover_ball(self, point_index=None):
        r""" Find a ball that covers the entire PointCloud.

        Parameters
        ----------
        point_index : int
            If point_index is given, we use that point as the center.
            If point_index is None (default), then we compute the point neareast
            the center-of-mass, which requires an extra :func:`numpy.mean` and
            :func:`scipy.spatial.distance.cdist` call.

        Returns
        -------
        ball : dict with keys 'index' (index of designated center point),
        'point' (coordinates of designated center point), and 'radius' (radius
        of ball)

        """
        if point_index is None:
            center = self.coords.values.mean(axis=0)
            center.shape = (1, center.shape[0])  # re-shape for cdist
            center_dists = cdist(self.coords.values, center, metric=self.dist)
            point_index = center_dists.argmin()

        point = np.array([point_index])
        indices = self.coords.index.values
        point_dists = self.dists(point, indices).flatten()
        return {'index': point_index,
                'point': self.coords.values[point_index, :],
                'radius': point_dists.max()}

    @classmethod
    def from_multisample_multilabel(cls, list_of_samples, list_of_labels,
                                    equal_priors=True, normalize_domain=False):
        r""" Produce a single labeled and weighted pointcloud from a list of samples
        and labels of those samples.

        Parameters
        ----------
        list_of_samples :
            A list (or np array) of np arrays.  Each such array is
            considered to be a sample of N points in R^d.  N can vary between
            entries, but d cannot.

        list_of_labels :
            A list of labels.  Labels can be anything, but it is covenient to
            use strings like "red" and "blue".    list_of_labels[i] is the
            label for the points in list_of_samples[i].

        equal_priors:
            Re-normalize weights so that each label is equally likely.
            Default: True

        normalize_domain:
            Use SVD/PCA to re-shape the original data to be roughy spherical.
            This should allow better learning via CDER.
            Default: False

        """
        assert len(list_of_labels) == len(list_of_samples),\
            "list_of_labels must equal list_of arrays. {} != {}".format(
                len(list_of_labels), len(list_of_samples))

        ambient_dim = list_of_samples[0].shape[1]
        assert all([X.shape[1] == ambient_dim for X in list_of_samples]),\
            "Dimension mismatch among list_of_samples!"

        label_info = pd.DataFrame(index=sorted(list(set(list(list_of_labels)))))
        # ['blue', 'green', 'red']
        # label_names = list(label_dict.keys())
        # label_index = np.array
        # num_labels = len(label_dict)

        # count how many times each label occurs.
        label_info['clouds'] = np.zeros(shape = (len(label_info),), dtype=np.int64)
        label_info['points'] = np.zeros(shape = (len(label_info),), dtype=np.int64)
        label_info['weight'] = np.zeros(shape = (len(label_info),), dtype=np.float64)
        for label in label_info.index:
            label_bool = np.array([l == label for l in list_of_labels])
            label_info.loc[label, 'clouds'] = np.count_nonzero(label_bool)
        label_info['int_index'] = label_info.index.get_indexer(label_info.index)

        # merge the samples into one big dataset
        # expand the sample-wise labels to point-wise labels
        points = np.concatenate(list(list_of_samples))

        if normalize_domain:
            m,s,v = fast_algorithms.gaussian_fit(points)
            points = np.dot((points - m), v.T)/s

        pointwise_labels = []  # keep track of labels, pointwise
        pointwise_source = []
        pointwise_weight = []
        for i, X in enumerate(list_of_samples):
            l = list_of_labels[i]
            num = X.shape[0]
            assert num > 0, "bad? {}".format(X.shape)
            pointwise_labels.extend([label_info['int_index'].loc[l]]*num)
            pointwise_source.extend([i]*num)
            if equal_priors:
                wt = 1.0/num/label_info.loc[l, 'clouds']
            else:
                wt = 1.0
            pointwise_weight.extend([wt]*num)
            label_info.loc[l, 'points'] = np.int64(label_info.loc[l, 'points']) + np.int64(num)
            label_info.loc[l, 'weight'] += wt*num

        pointwise_labels = np.array(pointwise_labels)
        pointwise_source = np.array(pointwise_source)
        pointwise_weight = np.array(pointwise_weight)

        pc = cls(points, masses=pointwise_weight)
        pc.label_info = label_info
        pc.labels = pointwise_labels
        pc.source = pointwise_source
        return pc

