r"""
This module defines tools for geometric analysis of one-dimensional
(time-series) data sets.  The main classes are

    - :class:`Signal`
    - :class:`SpaceCurve`

See `timeseries-data` for a more general outline.

Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE

Examples
--------

>>> el = Signal([2.0, 3.0, 0.0, 5.0, 2.5, 2.9])
>>> el.make_pers()
>>> el.pers.diagram
   birth_index  death_index  birth  death  pers
0            0            1    2.0    3.0   1.0
1            4            3    2.5    5.0   2.5
2            2            3    0.0    5.0   5.0
>>> bin_counts, bins = el.pers.bin(1.0)
>>> print(bins)
[-inf   0.   1.   2.   3.   4.   5.  inf]
>>> print(bin_counts)
[[ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  1.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  1.  1.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.]]
>>> sorted(list(el.pers.domains.keys()))
[(0, 1), (2, 3), (3, 4)]
>>> for interval in el.iter_intervals():
...     print("h_A( {} ) == {}".format(interval,el.interval_height(interval)))
h_A( (0, 1) ) == 1.0
h_A( (0, 2) ) == 0.0
h_A( (0, 3) ) == 0.0
h_A( (0, 4) ) == 0.0
h_A( (0, 5) ) == 0.0
h_A( (1, 2) ) == 0.0
h_A( (1, 3) ) == 0.0
h_A( (1, 4) ) == 0.0
h_A( (1, 5) ) == 0.0
h_A( (2, 3) ) == 5.0
h_A( (2, 4) ) == 0.0
h_A( (2, 5) ) == 0.0
h_A( (3, 4) ) == 2.5
h_A( (3, 5) ) == 0.0
h_A( (4, 5) ) == 0.0
>>> list(el.pers.forest.keys())
[None]
>>> sorted(list(el.pers.forest[None]))
[0, 1, 2]
>>> el.jagged(6.0)
0    0.0
1    0.0
2    0.0
3    0.0
4    0.0
5    0.0
dtype: float64
>>> el = Signal([6.5,0.0,2.0])
>>> el.make_pers()
>>> el.pers.diagram
   birth_index  death_index  birth  death  pers
0            1            0    0.0    6.5   6.5
>>> list(el.pers.forest[None])
[0]
>>> el.vertices
   time  height
0   0.0     6.5
1   1.0     0.0
2   2.0     2.0
>>> el.jagged() #el.pers.diagram,el.vertices.index,el.vertices.dtype)
0    6.5
1    0.0
2    0.0
dtype: float64
>>> elN = el.sample_near(sigma=0.1)
>>> elN.make_pers()
>>> elN.pers.domains == el.pers.domains
True
>>> el = Signal([6.5,7.0,2.0,4.5,3.9,9.0,8.3,8.7,5.5,9.9])
>>> el.vertices
   time  height
0   0.0     6.5
1   1.0     7.0
2   2.0     2.0
3   3.0     4.5
4   4.0     3.9
5   5.0     9.0
6   6.0     8.3
7   7.0     8.7
8   8.0     5.5
9   9.0     9.9
>>> el.edges
   src  dst  max  height
2    2    3    3     4.5
3    3    4    3     4.5
0    0    1    1     7.0
1    1    2    1     7.0
6    6    7    7     8.7
7    7    8    7     8.7
4    4    5    5     9.0
5    5    6    5     9.0
8    8    9    9     9.9
>>> el.make_pers()
>>> el.pers.diagram
   birth_index  death_index  birth  death  pers
2            6            7    8.3    8.7   0.4
1            0            1    6.5    7.0   0.5
0            4            3    3.9    4.5   0.6
3            8            5    5.5    9.0   3.5
4            2            9    2.0    9.9   7.9
>>> el.feature()
   time  height
2   2.0     2.0
3   3.0     4.5
4   4.0     3.9
5   5.0     9.0
6   6.0     8.3
7   7.0     8.7
8   8.0     5.5
9   9.0     9.9
>>> el.pers.syzygy((0,0))
2    0.4
1    0.5
0    0.6
3    3.5
4    7.9
dtype: float64
>>> el.pers.syzygy((1,2))
2     251.2908
1     159.2500
0      47.3850
3    1559.2500
4    1548.5580
dtype: float64
>>> el.pers.forest_parents == {0: 4, 1: None, 2: 3, 3: 4, 4: None}
True
>>> el.vertices['height'].sum()/len(el.vertices)
6.5299999999999994
>>> el.normalize()
>>> el2 = Signal(el.vertices)
>>> el2.make_pers()
>>> all(el2.pers.diagram == el.pers.diagram)
True
>>> el = Signal([0.0,3.0,1.5,2.2,0.001])
>>> el.make_pers()
>>> el.vertices
   time  height
0   0.0   0.000
1   1.0   3.000
2   2.0   1.500
3   3.0   2.200
4   4.0   0.001
>>> el.edges
   src  dst  max  height
2    2    3    3     2.2
3    3    4    3     2.2
0    0    1    1     3.0
1    1    2    1     3.0
>>> el.pers.diagram
   birth_index  death_index  birth  death   pers
0            2            3  1.500    2.2  0.700
1            4            1  0.001    3.0  2.999
2            0            1  0.000    3.0  3.000
>>> el = Signal([0.0,0.7,0.45,0.55,0.3, 1.0],
...             times=[0.1, 0.2, 0.3, 0.6, 0.8, 0.85])
>>> el.vertices
   time  height
0  0.10    0.00
1  0.20    0.70
2  0.30    0.45
3  0.60    0.55
4  0.80    0.30
5  0.85    1.00
>>> el.edges
   src  dst  max  height
2    2    3    3    0.55
3    3    4    3    0.55
0    0    1    1    0.70
1    1    2    1    0.70
4    4    5    5    1.00
>>> el.make_pers()
>>> el.pers.diagram
   birth_index  death_index  birth  death  pers
0            2            3   0.45   0.55   0.1
1            4            1   0.30   0.70   0.4
2            0            5   0.00   1.00   1.0
>>> el = Signal([0.0,0.5,0.4,0.9,0.1,1.0])
>>> el.make_pers()
>>> el.pers.diagram
   birth_index  death_index  birth  death  pers
0            2            1    0.4    0.5   0.1
1            4            3    0.1    0.9   0.8
2            0            5    0.0    1.0   1.0
>>> for F in el.iter_features(min_pers=0.5): print(F)
   time  height
0   0.0     0.0
1   1.0     0.5
2   2.0     0.4
3   3.0     0.9
4   4.0     0.1
5   5.0     1.0
   time  height
3   3.0     0.9
4   4.0     0.1
>>> el = Signal(np.sin(np.arange(0,8*np.pi,0.1)))
>>> el.make_pers()
>>> el.pers.domains == {(110, 204): 4, (204, 236): 3, (0, 16): 2, (47, 79): 0, (141, 173): 1}
True


>>> s = Signal([50.0, 120.0, 100, 180, 200, 150, 135])
>>> s.make_pers()
>>> s.pers.diagram
   birth_index  death_index  birth  death   pers
0            2            1  100.0  120.0   20.0
1            6            4  135.0  200.0   65.0
2            0            4   50.0  200.0  150.0
>>> s.pers.mergetree
{1: (0, 2), 4: (1, 6)}
"""

import collections, itertools

import numpy as np
import pandas as pd

import scipy.spatial.distance as ssd
from . import timeseries_fast_algorithms, curve_geometry
import homology.dim0


def jagged(persdiag, index):
    """ Produce a piecewise-linear function that matches the given persistence
    diagram. This assumes that the index for the vertices is sequential and
    linear, so that the mergetree can be ignored.
    
    Parameters
    ----------
    persdiag : :class:`homology.PersDiag`
        A 0-dimensional persistence diagram
    index : list-like
        The domain index for the function

    Returns
    -------
    function : `pandas.Series`

    See Also
    --------
    :func:`timeseries.Signal.jagged` :func:`timeseries.Signal.makepers`
    """
    V = pd.Series(index=index, dtype=np.float64)
    if len(persdiag) == 0:
        V[:] = 0
    V[persdiag['birth_index']] = persdiag['birth']
    V[persdiag['death_index']] = persdiag['death']
    # fill with linear interpolation
    V.interpolate(method='linear', inplace=True) 
    # make sure we don't lose domain
    V.fillna(method='ffill', inplace=True)  # the end
    V.fillna(method='bfill', inplace=True)  # the beginning
    # MUST ADD CIRCULAR INTERPOLATION!
    return V

def wavy(pasrdiad, index):
    r""" Produce a piecewise-sine function that matches the given persistence
    diagram. This assumes that the index for the vertices is sequential and
    linear, so that the mergetree can be ignored.

    Parameters
    ----------
    persdiag : :class:`homology.PersDiag`
        A 0-dimensional persistence diagram
    index : list-like
        The domain index for the function

    Returns
    -------
    function : `pandas.Series`

    See Also
    --------
    :func:`timeseries.Signal.jagged` :func:`timeseries.Signal.makepers`
    """
    #V = pd.Series(index=index, dtype=np.float64)
    #if len(persdiag) == 0:
    #    V[:] = 0
    #V[persdiag['birth_index']] = persdiag['birth']
    #V[persdiag['death_index']] = persdiag['death']
    ## fill with linear interpolation
    #V.interpolate(method='linear', inplace=True) 
    ## make sure we don't lose domain
    #V.fillna(method='ffill', inplace=True)  # the end
    #V.fillna(method='bfill', inplace=True)  # the beginning
    # MUST ADD CIRCULAR INTERPOLATION!
    #return V

class Signal(object):

    def __init__(self, values, times=None):
        """ produce an Signal from function output values.  function input
        domain is implicitly range(len(values)).  The resulting Signal has
        vertices with function values and edges with max of adjacent pairs.
            A Signal is a pair of pd DataFrames that act as indexed
        lists of numerical values.  The vertices are stored as a pd Series,
        Signal.index provides the vertex indices.
            The edges are a DataFrame, giving triples labeled by
        ('src', 'dst', 'max', 'height'), where src and dst are the *indices* (not
        the *values*) of vertices.  Vertices should be considered fixed on
        initialization.  Overloaded functions refer to edges, not vertices.
        """

        if type(values) == pd.core.frame.DataFrame:
            assert values.index.dtype == np.int64
            assert all(values.columns == ['time', 'height'])
            assert values['height'].dtype == np.float64
            assert values['time'].dtype == np.float64
            self.vertices = values.copy()
        else:
 
            values = np.array(values, dtype=np.float64)
            assert len(values.shape) == 1
        
            N = values.shape[0]

            if times is None:
                times = np.arange(N, dtype=np.float64)

            index = np.arange(N, dtype=np.int64)

            self.vertices = pd.DataFrame({
                'time': times,
                'height': values},
                columns=['time', 'height'],
                index=index)

        # if times is not None:
        #     self.times
        #
        # if type(values) == pd.core.series.Series and values.index.dtype == 'int64':
        #     self.vertices = values.copy()
        # else:
        #     self.vertices = pd.Series(values, dtype=np.float64,
        #         index=np.arange(len(values), dtype=np.int64))

        # figure out whether the value is from the left (0) or right (1)
        # this is now done in homology.dim0.unionfind.
        leftright = np.array([self.vertices['height'].values[:-1],
                              self.vertices['height'].values[1:]]).argmax(axis=0)
        maxes = pd.Series(self.vertices.index[:-1] + leftright, dtype=np.int64)
        edges = pd.DataFrame({
                'src': self.vertices['height'].index[:-1],
                'dst': self.vertices['height'].index[1:],
                'max': maxes.values,
                'height':  self.vertices['height'].values[maxes]},
                columns=['src', 'dst', 'max', 'height'])
        
        self.edges = pd.DataFrame(edges)
        self.edges.sort_values(by=['height', 'src'],
                               ascending=[True, True],
                               inplace=True)

        self.components=pd.Series(self.vertices.index,
                                  index=self.vertices.index)

        self.pers = None
        # assert pos.size == len(self.edges.index)
        # self.edges['pos'] = pos

    def make_pers(self, show_diagonal=False, cutoff=-1.0, until_connected=(0,0)):
        tbirth_index, tdeath_index, ybirth_index, ydeath_index, mergetree = homology.dim0.unionfind(self,
            np.float64(cutoff), np.int64(show_diagonal), np.int64(until_connected[0]), np.int64(until_connected[1]))
        self.pers = homology.PersDiag(tbirth_index, tdeath_index, ybirth_index, ydeath_index, mergetree)


    def __len__(self):
        return len(self.vertices)

    def gap(self):
        """ Return the largest homology. """
        bc = self.pers.diagram
        if len(bc) > 0:
            return bc.loc[bc.index[-1]]  # sorted by length!
        else: 
            return bc

    def iter_intervals(self):
        """ return the itertools combinations iterator over all sub-intervals.
        """
        return itertools.combinations(self.vertices.index.values, 2)

    def interval_height(self, interval):
        """ the indicator-persistence function for intervals,
        called h_U(A) in the notes.
        """
        if interval in self.pers.domains:
            index = self.pers.domains[interval]
            return self.pers.diagram['pers'].loc[index]
        return 0.0

    def sample_near(self, sigma=1.0):
        """ return a Signal object that is L2-near self in the normal
        distribution. 
        """
        diff = np.random.randn(self.vertices['height'].values.shape[0])
        return self.__class__(self.vertices['height'].values + sigma*diff) 

    def height_measure(self, sigma=1.0, num_samples=1000, parallel=True, min_pers=0):
        """ Use a simulation to estimate the height-measure of an interval. """
        all_data = [(self, sigma) for _ in range(num_samples)]

        if parallel:
            from concurrent.futures import ProcessPoolExecutor
            pool = ProcessPoolExecutor(max_workers=None) 
            all_heights = list(pool.map(fast_algorithms.sample_height, all_data))
        else:
            all_heights = [fast_algorithms.sample_height(x) for x in all_data]
        all_heights = np.stack(all_heights)
        if min_pers > 0:
            all_heights[all_heights < min_pers] = 0
        tot_heights = all_heights.sum(axis=0)/num_samples
        measures = dict()
        for U in np.stack(tot_heights.nonzero()).T:
            i, j = U
            length = j-i+1
            if length not in measures:
                measures[length] = dict()
            measures[length][i] = tot_heights[i, j]
        return measures

    def feature(self, index=None):
        """ return the region of a feature, with vertical displacement threshold tau. """
        if index is None:
            index = self.gap().name
        left = self.pers.lefts[index]
        right = self.pers.rights[index]
        V = self.vertices[left:right+1]
        return V

    def iter_features(self, min_pers=0, depth_first=False):
        """ walk the feature tree. """
        bc = self.pers.diagram
        tree = self.pers.forest
        to_walk = collections.deque(tree[None])  # start at root of tree
        while to_walk:
            idx = to_walk.popleft()
            if bc.loc[idx]['pers'] > min_pers:
                yield self.feature(index=idx) 
            if depth_first:
                to_walk.extendleft(tree[idx])
            else:
                to_walk.extend(tree[idx])

    def feature_match(self, other, sigma=1.0, num_samples=1000):
        total_match = 0
        for left, right in self.iter_intervals():
            feature = self.vertices[left:right+1].values
            match_number = other.profile(feature).max()
            interval_height = self.height_measure((left, right),
                                                  sigma=sigma,
                                                  num_samples=num_samples)
            total_match += match_number * interval_height

        return total_match

    def jagged(self, beta=0):
        """ call :func:`timeseries.jagged` on this :class:`Signal`'s 
        own persistence diagram. This effectively makes a piecewise-linear
        version of the same function, with the same extrema. """
        # simple cache
        try: 
            self._jagged
            if beta in self._jagged:
                return self._jagged[beta]
        except AttributeError as e:
            self._jagged = dict()
        
        keepbc = self.pers.clip(beta)['keepcode']
        self._jagged[beta] = jagged(keepbc,
                                    self.vertices.index)
        return self._jagged[beta]

    def smoothen(self, beta):
        T = self.vertices['time'].values
        F = self.vertices['height'].values
        N = T.shape[0]
        pd = self.pers.clip(beta)['keepcode']
        cut_indices = np.concatenate([pd[['birth_index', 'death_index']].values.flatten(),
                                      np.array([0, N], dtype='int')])
        cut_indices = np.unique(cut_indices)

        times = [ ]
        segments = [ ]
    
        for j0,j1 in zip(cut_indices[:-1], cut_indices[1:]):        
            times.append(T[j0:j1])
            if F[j0] > F[min(j1,N-1)]:
                segments.append(np.sort(F[j0:j1])[::-1])
            elif F[j0] <= F[min(j1,N-1)]:
                segments.append(np.sort(F[j0:j1]))
            #else:
            #    assert F[j0:j1].min() == F[j0:j1].max()
    
        times = np.concatenate(times)
        segments = np.concatenate(segments)
        
        assert np.all(np.sort(segments) == np.sort(F))
        assert np.all(times == T)
        return Signal(segments, times=times)


    def profile(self, arch, normalize=False, norm=np.linalg.norm):
        """ produce profile by dragging an archetype across self, 
        looking for matches.  You may want to normalize arch first. 
        """
        a = len(arch)
        v = len(self.vertices)
        assert a <= v, "Archetype is too long."
        d = v - a + 1
#        if normalize: arch = (arch-arch.mean())/(arch.max() - arch.min())
        s = []
        for i in range(d):
            snip = self.vertices.values[i:a+i]
            if normalize:
                snip = (snip-snip.mean())/(snip.max() - snip.min())
            s.append(norm(arch - snip))
        p = np.exp2(-np.array(s))
        return p
        # if len(p) > 1:
        #    P = Signal(p)
        # return P

    def iter_windows_by_index(self, window, step=1, start=None, stop=None):
        """ Produce equal-length Signals using a sliding-window on self.  This
        slides by index, not by abstract time. 
        window = length of window (number of indices) 
        time_step  = step size in index.  (default = 1)
        time_start = start index (default = None, min index)
        time_stop  = stop index  (default = None, max index)
        normalize = renormalize by N(0,1) on window? (default = False)
        norm = norm function to use for comparison (default = np.linalg.norm)


        return: iterator of np arrays, whose columns are timestamp, value.
        To access the tracks as a list,use
        list(Signal.iter_windows_index())
        
        
        Examples
        --------
        >>> S = Signal([2.0,3.0,0.0,5.0,2.5,2.9])
        >>> for s in S.iter_windows_by_index(4, step=2):
        ...     print(s[:,1])
        [ 2.  3.  0.  5.]
        [ 0.   5.   2.5  2.9]
        """

        if start is None:
            start = self.vertices.index.values[0]
        if stop is None:
            stop = self.vertices.index.values[-1]

        slices = np.arange(start, stop, step)
        for i, start_i in enumerate(slices):
            stop_i = start_i + window
            yield self.vertices.values[start_i:stop_i, :]

            if stop_i >= stop:
                break
 
    def self_similarity(self, window, step=1, start=None, stop=None, 
                        dist=ssd.euclidean, normalizer=None):
        """ Compare sliding windows of this Signal using a distance function. 
       
        Parameters
        ----------
        window  (length of segment)
        step    (steps to move between windows)
        start   (index to start at)
        stop    (index to stop at)
        dist    (distance function to use. Default:`scipy.spatial.distance.euclidean`
        normalizer (function to use to renormalize each window.  default:None)
        
        Returns
        -------
        an iterator of the window comparisons.
        (0,0), (0,1), (0,2), ... (0, n-1), (1,1), (1,2), ...  (n-2, n-1)
        
        The return elements are pairs ((index_lo, index_hi), norm), which 
        can be used to populate a dictionary or array.

        Examples
        --------
        >>> S = Signal([0.0, 0.0, 3.0, 4.0, 6.0, 8.0])
        >>> Sss = list(S.self_similarity(window=2, step=2))
        >>> for (ij, d) in Sss:
        ...     print("{} -> {}".format(ij,d))
        (0, 0) -> 0.0
        (0, 2) -> 5.0
        (0, 4) -> 10.0
        (2, 2) -> 0.0
        (2, 4) -> 5.0
        (4, 4) -> 0.0
        >>> D = np.array([d for ij,d in Sss])
        >>> print(D)
        [  0.   5.  10.   0.   5.   0.]
        >>> print(ssd.squareform(D)) # scipy.spatial.distance
        [[  0.   0.   5.  10.]
         [  0.   0.   0.   5.]
         [  5.   0.   0.   0.]
         [ 10.   5.   0.   0.]]

        """
        if start is None:
            start = self.vertices.index.values[0]
        if stop is None:
            stop = self.vertices.index.values[-1]

        slices = np.arange(start, stop, step)
        for i, start_i in enumerate(slices):
            stop_i = start_i + window
            win_i = self.vertices.values[start_i:stop_i, :]
            if normalizer is not None:
                win_i = normalizer(win_i)
            for j, start_j in enumerate(slices[i:]):
                stop_j = start_j + window
                win_j = self.vertices.values[start_j:stop_j, :]
                if normalizer is not None:
                    win_j = normalizer(win_j)
                yield ((start_i, start_j), dist(win_i[:, 1], win_j[:, 1]))

            if stop_i >= stop:
                break
 
#
    def plot(self, canvas, title="Signal"):
        """ Plot the Signal.
        
        Parameters
        ----------
        canvas : class:`bokeh.plotting.figure.Figure` or :class:`matplotlib.axes._subplots.AxesSubplot`
                A bokeh or pyplot canvas to draw on.  Create one with
                :code:`canvas = bokeh.plotting.figure()` or
                :code:`fig,canvas = matplotlib.pyplot.subplots()`
        
        Notes
        -----
        You have to save or show axes after running this call.

        """
 
        if type(canvas).__module__ == 'bokeh.plotting.figure':
            canvas_type = "bokeh"
            import bokeh.plotting
        elif type(canvas).__module__ == 'matplotlib.axes._subplots':
            canvas_type = "pyplot"
            import matplotlib.pyplot as plt
        else:
            raise NotImplementedError(
                "canvas must be a bokeh.plotting.figure() or a matplotlib.pyplot.subplots()[1].  You gave me {}".format(type(canvas))
                ) 
        if canvas_type == "bokeh":
            canvas.title=title
        elif canvas_type == "pyplot":
            canvas.set_title(title)

        if canvas_type == "bokeh":
            canvas.circle(self.vertices['time'].values, self.vertices['height'].values)
            canvas.line(self.vertices['time'].values, self.vertices['height'].values)

        elif canvas_type == "pyplot":
            canvas.scatter(self.vertices['height'].index.values, self.vertices['height'].values)
            canvas.plot(self.vertices['height'].index.values, self.vertices['height'].values)
        pass

    @classmethod
    def from_pointcloud(cls, points, direction, norm):
        values = np.dot(points, direction/norm(direction))
        return cls(values)

    def normalize(self):
        """ change this Signal object to have mean = 0 and max-min = 1 """
        bc=self.pers.diagram
        h = self.gap()['pers']
        mean = self.vertices['height'].mean()
        self.vertices['height'] = (self.vertices['height'] - mean)/h
        self.edges['height'] = (self.edges['height'] - mean)/h
        bc['birth'] = (bc['birth'] - mean)/h
        bc['death'] = (bc['death'] - mean)/h
        bc['pers'] = (bc['pers'])/h
        pass


class SpaceCurve(object):
    r""" SpaceCurve is a Python class for studying curves in 
    :math:`\mathbb{R}^2` or :math:`\mathbb{R}^3`.
    For example, a SpaceCurve could represent kinematic flight data, or
    trajectories of vehicles given by GPS coordinates.
    
    All arguments other than :code:`tn` are optional.

    Parameters
    ----------
    tn : list-like
        Integer timestamps, typically in 'numpy.int64` nanoseconds-since-epoch
    px : list-like
    py : list-like
    pz : list-like
        Positions in :class:`numpy.float64` meters
    quality : list-like
        Quality/accuracy of a particular location
    trackid : int
        An integer label for the track
    platform : str
        A descriptive label
    activity : str
        A descriptive label
    mollified: bool
        Whether this track has undegone mollification, for example with :func:`clean_copy`
    
    
    Attributes
    ----------
    data : :class:`pandas.DataFrame`
        The original position and velocity data, as originally provided.
        The index of this DataFrame is the :code:`tn` integer time index.

    info : :class:`pandas.DataFrame`
        Data computed using various algorithms.  This is filled by 
        :func:`compute`, but more can be added.


    """

    def __init__(self, tn, px=None, py=None, pz=None,
    #             vx=None, vy=None, vz=None,
                 quality=None,
                 trackid=-1, platform=None, activity=None, mollified=False):
        tn = np.array(tn, dtype=np.int64)
        
        assert len(tn) > 0,\
            "A SpaceCurve cannot have empty nanosecond index. You gave me {}".format(tn) 

        if px is None:
            px = np.zeros(tn.shape, dtype=np.float64)
        if py is None:
            py = np.zeros(tn.shape, dtype=np.float64)
        if pz is None:
            pz = np.zeros(tn.shape, dtype=np.float64)
        #if vx is None:
        #    vx = np.zeros(tn.shape, dtype=np.float64)
        #if vy is None:
        #    vy = np.zeros(tn.shape, dtype=np.float64)
        #if vz is None:
        #    vz = np.zeros(tn.shape, dtype=np.float64)
        if quality is None:
            quality = -np.ones(tn.shape, dtype=np.int64)
        px = np.array(px, dtype=np.float64)
        py = np.array(py, dtype=np.float64)
        pz = np.array(pz, dtype=np.float64)
        #vx = np.array(vx, dtype=np.float64)
        #vy = np.array(vy, dtype=np.float64)
        #vz = np.array(vz, dtype=np.float64)
        quality = np.array(quality, dtype=np.int64)

        assert len(tn) == len(px)
        assert len(tn) == len(py)
        assert len(tn) == len(pz)
        #assert len(tn) == len(vx)
        #assert len(tn) == len(vy)
        #assert len(tn) == len(vz)
        assert len(tn) == len(quality)

        sort_by_time = tn.argsort()
        
        tn = tn[sort_by_time]
        px = px[sort_by_time]
        py = py[sort_by_time]
        pz = pz[sort_by_time]
        #vx = vx[sort_by_time]
        #vy = vy[sort_by_time]
        #vz = vz[sort_by_time]
        quality = quality[sort_by_time]

        ts = (tn - tn[0]).astype(np.float64) / (10 ** 9)

        self.data = pd.DataFrame({'time': ts,
                                  'pos_x': px,
                                  'pos_y': py,
                                  'pos_z': pz,
                                  #'vel_x': vx,
                                  #'vel_y': vy,
                                  #'vel_z': vz,
                                  'quality': quality},
                                 columns=['time', 'pos_x', 'pos_y', 'pos_z', 
                                                  #'vel_x', 'vel_y', 'vel_z',
                                                  'quality'],
                                 index=tn)

        self.info = pd.DataFrame({}, index=self.data.index)
        self.trackid = trackid
        self.platform = platform
        self.activity = activity
        self.mollified = mollified

    def __getitem__(self, key):
        """ get raw data via index """
        return self.data.loc[self.data.index[key]]
    
    def accel(self, rate):
        r"""Change time parametrization, to represent a constant tangential 
        acceleration (or deceleration).  Locations, initial time, and
        arc-length are preserved.
        
        The first timestep is changed to have (1-rate) times the speed of the
        original's first timestep.  The last timestep is changed to have
        (1+rate) times the speed of the original's last timestep.  
        That is, if S is a `SpaceCurve` of constant speed, then S.accel(-0.1)
        will start 10% faster and end 10% slower than S. 


        If speed[i] changes to speed[i]*q[i], then delta_t[i] changes to 
        delta_t[i]*p[i], where p[i] = 1 / q[i]

        Examples
        --------
        >>> tn = np.arange(0, 5*1e9, 1e9)
        >>> s = SpaceCurve(tn=tn, px=10.0*tn/1e9) # drive straight at 10m/s
        >>> s
        SpaceCurve with 5 entries and duration 4.000000000
        >>> s.data
                    time  pos_x  pos_y  pos_z  quality
        0            0.0    0.0    0.0    0.0       -1
        1000000000   1.0   10.0    0.0    0.0       -1
        2000000000   2.0   20.0    0.0    0.0       -1
        3000000000   3.0   30.0    0.0    0.0       -1
        4000000000   4.0   40.0    0.0    0.0       -1
        >>> s.compute()
        >>> s.info['speed']
        0             10.0
        1000000000    10.0
        2000000000    10.0
        3000000000    10.0
        4000000000    10.0
        Name: speed, dtype: float64
        >>> a = s.accel(0.25)
        >>> a
        SpaceCurve with 5 entries and duration 4.147319347
        >>> a.compute()
        >>> a.info['speed']
        0              7.500000
        1333333333     8.250000
        2424242424     9.930556
        3347319347    11.607143
        4147319347    12.500000
        Name: speed, dtype: float64
        >>> a.data
                        time  pos_x  pos_y  pos_z  quality
        0           0.000000    0.0    0.0    0.0       -1
        1333333333  1.333333   10.0    0.0    0.0       -1
        2424242424  2.424242   20.0    0.0    0.0       -1
        3347319347  3.347319   30.0    0.0    0.0       -1
        4147319347  4.147319   40.0    0.0    0.0       -1
        >>> b = s.accel(-0.25) 
        >>> b
        SpaceCurve with 5 entries and duration 4.147319347
        >>> b.compute()
        >>> b.info['speed']
        0             12.500000
        800000000     11.607143
        1723076923     9.930556
        2813986013     8.250000
        4147319347     7.500000
        Name: speed, dtype: float64
        >>> b.data
                        time  pos_x  pos_y  pos_z  quality
        0           0.000000    0.0    0.0    0.0       -1
        800000000   0.800000   10.0    0.0    0.0       -1
        1723076923  1.723077   20.0    0.0    0.0       -1
        2813986013  2.813986   30.0    0.0    0.0       -1
        4147319347  4.147319   40.0    0.0    0.0       -1
        """

        n = len(self.data)
        ts = self.data['time'].values

        change_of_speed = np.linspace(1.-rate, 1.+rate, num=n-1)
        new_Dt = np.diff(ts)/change_of_speed
        accum_time = np.cumsum(new_Dt)
        
        new_ts = np.ndarray(shape=ts.shape, dtype=ts.dtype)
        new_ts[0] = ts[0]
        new_ts[1:] = accum_time + ts[0]
        
        new_tn = np.int64(new_ts*10**9)
        return SpaceCurve(new_tn,
                          px=self.data['pos_x'].values,
                          py=self.data['pos_y'].values,
                          pz=self.data['pos_z'].values)

    def __matmul__(self, array):
        """ Apply a matrix (NumPy array) to the positions to produce a new 
        SpaceCurve.  Used for rotating SpaceCurves.
        Note that this is the LEFT action from a group-theory perspective.

        Examples
        --------
        
        >>> sc = SpaceCurve(np.arange(4)*10**9, px=np.arange(4))
        >>> sc.data
                    time  pos_x  pos_y  pos_z  quality
        0            0.0    0.0    0.0    0.0       -1
        1000000000   1.0    1.0    0.0    0.0       -1
        2000000000   2.0    2.0    0.0    0.0       -1
        3000000000   3.0    3.0    0.0    0.0       -1
        >>> g = np.array([[np.cos(np.pi/6), -np.sin(np.pi/6), 0],
        ...               [np.sin(np.pi/6),  np.cos(np.pi/6), 0],
        ...               [         0,           0, 1]])
        >>> sc2 = sc.__matmul__(g).__matmul__(g).__matmul__(g) # use @ in Py3
        >>> np.all(sc2.data['pos_y'].values == sc.data['pos_x'].values)
        True
        """
       
        pos = self.data[['pos_x', 'pos_y', 'pos_z']].values
        new_pos = pos.dot(array.T)
        return SpaceCurve(self.data.index,
                          px=new_pos[:, 0],
                          py=new_pos[:, 1],
                          pz=new_pos[:, 2])

    def __add__(self, other):
        """ Concatenate SpaceCurves, end-to-end in space and time. The other
        SpaceCurve is set to begin at a time and position where self ends.
        This way, the number of points shrinks by one, but the total duration
        adds.
        
        Examples
        --------
        
        >>> sc1 = SpaceCurve(np.arange(4)*10**9, px=np.arange(4), py=2*np.arange(4))
        >>> sc1
        SpaceCurve with 4 entries and duration 3.000000000
        >>> sc1.data
                    time  pos_x  pos_y  pos_z  quality
        0            0.0    0.0    0.0    0.0       -1
        1000000000   1.0    1.0    2.0    0.0       -1
        2000000000   2.0    2.0    4.0    0.0       -1
        3000000000   3.0    3.0    6.0    0.0       -1
        >>> sc2 = SpaceCurve(np.arange(8)*10**8, px=np.arange(0,0.4,0.05), py=3*np.arange(8))
        >>> sc2
        SpaceCurve with 8 entries and duration .700000000
        >>> sc2.data
                   time  pos_x  pos_y  pos_z  quality
        0           0.0   0.00    0.0    0.0       -1
        100000000   0.1   0.05    3.0    0.0       -1
        200000000   0.2   0.10    6.0    0.0       -1
        300000000   0.3   0.15    9.0    0.0       -1
        400000000   0.4   0.20   12.0    0.0       -1
        500000000   0.5   0.25   15.0    0.0       -1
        600000000   0.6   0.30   18.0    0.0       -1
        700000000   0.7   0.35   21.0    0.0       -1
        >>> sc3 = sc1 + sc2
        >>> sc3
        SpaceCurve with 11 entries and duration 3.700000000
        >>> sc3.data
                    time  pos_x  pos_y  pos_z  quality
        0            0.0   0.00    0.0    0.0       -1
        1000000000   1.0   1.00    2.0    0.0       -1
        2000000000   2.0   2.00    4.0    0.0       -1
        3000000000   3.0   3.00    6.0    0.0       -1
        3100000000   3.1   3.05    9.0    0.0       -1
        3200000000   3.2   3.10   12.0    0.0       -1
        3300000000   3.3   3.15   15.0    0.0       -1
        3400000000   3.4   3.20   18.0    0.0       -1
        3500000000   3.5   3.25   21.0    0.0       -1
        3600000000   3.6   3.30   24.0    0.0       -1
        3700000000   3.7   3.35   27.0    0.0       -1
        """
        
        tn_shift = self.data.index.values[-1] - other.data.index.values[0]
        px_shift = self.data['pos_x'].values[-1] - other.data['pos_x'].values[0]
        py_shift = self.data['pos_y'].values[-1] - other.data['pos_y'].values[0]
        pz_shift = self.data['pos_z'].values[-1] - other.data['pos_z'].values[0]

        new_tn = np.concatenate([self.data.index.values,
                                 other.data.index.values[1:] + tn_shift])
        new_px = np.concatenate([self.data['pos_x'].values,
                                 other.data['pos_x'].values[1:] + px_shift])
        new_py = np.concatenate([self.data['pos_y'].values,
                                 other.data['pos_y'].values[1:] + py_shift])
        new_pz = np.concatenate([self.data['pos_z'].values,
                                 other.data['pos_z'].values[1:] + pz_shift])
        return self.__class__(new_tn, px=new_px, py=new_py, pz=new_pz)
 
    def arclength_param(self):
        """ Change time parametrization to the universal speed=1 arclength
        parametrization.


        Examples
        --------
        >>> tn = np.arange(0, 5e9, 1e9)
        >>> s = SpaceCurve(tn=tn, px=(tn/1e9)**2)
        >>> s.data
                    time  pos_x  pos_y  pos_z  quality
        0            0.0    0.0    0.0    0.0       -1
        1000000000   1.0    1.0    0.0    0.0       -1
        2000000000   2.0    4.0    0.0    0.0       -1
        3000000000   3.0    9.0    0.0    0.0       -1
        4000000000   4.0   16.0    0.0    0.0       -1
        >>> a = s.arclength_param()
        >>> a.data
                     time  pos_x  pos_y  pos_z  quality
        0             0.0    0.0    0.0    0.0       -1
        1000000000    1.0    1.0    0.0    0.0       -1
        4000000000    4.0    4.0    0.0    0.0       -1
        9000000000    9.0    9.0    0.0    0.0       -1
        16000000000  16.0   16.0    0.0    0.0       -1
        """

        pos = self.data[['pos_x', 'pos_y', 'pos_z']].values
        gap = np.diff(pos, axis=0)
        dist = np.linalg.norm(gap, axis=1)
        accum_time = dist.cumsum()

        ts = self.data['time'].values
        new_ts = np.ndarray(shape=ts.shape, dtype=ts.dtype)
        new_ts[0] = ts[0]
        new_ts[1:] = accum_time + ts[0]
        new_tn = np.int64(new_ts*10**9)
        return SpaceCurve(new_tn,
                          px=self.data['pos_x'].values,
                          py=self.data['pos_y'].values,
                          pz=self.data['pos_z'].values)

    def reverse(self):
        """
        Reverse the time parametrization of the SpaceCurve.

        Examples
        --------

        >>> tn = np.arange(0, 5*1e9, 1e9)
        >>> s = SpaceCurve(tn=tn, px=10.0*tn/1e9) # drive straight at 10m/s
        >>> s.data
                    time  pos_x  pos_y  pos_z  quality
        0            0.0    0.0    0.0    0.0       -1
        1000000000   1.0   10.0    0.0    0.0       -1
        2000000000   2.0   20.0    0.0    0.0       -1
        3000000000   3.0   30.0    0.0    0.0       -1
        4000000000   4.0   40.0    0.0    0.0       -1
        >>> a = s.reverse()
        >>> a.data
                    time  pos_x  pos_y  pos_z  quality
        0            0.0   40.0    0.0    0.0       -1
        1000000000   1.0   30.0    0.0    0.0       -1
        2000000000   2.0   20.0    0.0    0.0       -1
        3000000000   3.0   10.0    0.0    0.0       -1
        4000000000   4.0    0.0    0.0    0.0       -1
        """
        return SpaceCurve(self.data.index.values,
                          px=self.data['pos_x'].values[::-1],
                          py=self.data['pos_y'].values[::-1],
                          pz=self.data['pos_z'].values[::-1])

    def reparam(self, rate):
        r"""Change time parametrization, to represent a constant change of speed.
        Locations, initial time, and arc-length are preserved.
        
        Parameters
        ----------
        rate : float
            A (positive) ratio by which to increase the speed.

        Notes
        -----
        If :code:`speed[i]` changes to :code:`speed[i]*q[i]`, then
        :code:`delta_t[i]` changes to :code:`delta_t[i]*p[i]`, where 
        :code:`p[i] = 1 / q[i]`.

        Examples
        --------
        >>> tn = np.arange(0, 5*1e9, 1e9)
        >>> s = SpaceCurve(tn=tn, px=10.0*tn/1e9) # drive straight at 10m/s
        >>> s
        SpaceCurve with 5 entries and duration 4.000000000
        >>> s.data
                    time  pos_x  pos_y  pos_z  quality
        0            0.0    0.0    0.0    0.0       -1
        1000000000   1.0   10.0    0.0    0.0       -1
        2000000000   2.0   20.0    0.0    0.0       -1
        3000000000   3.0   30.0    0.0    0.0       -1
        4000000000   4.0   40.0    0.0    0.0       -1
        >>> a = s.reparam(0.5)
        >>> a.data
                    time  pos_x  pos_y  pos_z  quality
        0            0.0    0.0    0.0    0.0       -1
        2000000000   2.0   10.0    0.0    0.0       -1
        4000000000   4.0   20.0    0.0    0.0       -1
        6000000000   6.0   30.0    0.0    0.0       -1
        8000000000   8.0   40.0    0.0    0.0       -1
        >>> a
        SpaceCurve with 5 entries and duration 8.000000000
        >>> b = s.reparam(2.0) 
        >>> b
        SpaceCurve with 5 entries and duration 2.000000000
        >>> b.data
                    time  pos_x  pos_y  pos_z  quality
        0            0.0    0.0    0.0    0.0       -1
        500000000    0.5   10.0    0.0    0.0       -1
        1000000000   1.0   20.0    0.0    0.0       -1
        1500000000   1.5   30.0    0.0    0.0       -1
        2000000000   2.0   40.0    0.0    0.0       -1
        """

        ts = self.data['time'].values
        new_ts = ts/rate
        new_tn = np.int64(new_ts*10**9)
        return SpaceCurve(new_tn,
                          px=self.data['pos_x'].values,
                          py=self.data['pos_y'].values,
                          pz=self.data['pos_z'].values)

    def plot(self, canvas, title="SpaceCurve", color="blue"):
        r""" Plot the SpaceCurve in 3D.
        
        Parameters
        -----------
        canvas : :class:`matplotlib.axes._subplots.AxesSubplot`
                Be sure that 3D plotting is enabled on this canvas, with
                :code:`mpl_toolkits.mplot3d import Axes3D` and 
                :code:`canvas = matplotlib.pyplot.subplot(projection='3d')`
        title : str
                A title for the figure.
        color : str
                The name of a color for the points to draw.  Passed to the
                appropriate drawing library (bokeh or matplotlib).


        Notes
        -----
        Because we are using ECEF coordinates, the horizontal projection should
        be taken as an approximation!
        """

        if type(canvas).__module__ == 'bokeh.plotting.figure':
            canvas_type = "bokeh"
            import bokeh.plotting
        elif type(canvas).__module__ == 'matplotlib.axes._subplots':
            canvas_type = "pyplot"
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        else:
            raise NotImplementedError(
                "canvas must be a bokeh.plotting.figure() or a matplotlib.pyplot.subplots()[1].  You gave me {}".format(
                    type(canvas))
            )

        #if canvas_type == "bokeh":
        #    canvas.title = title
        #elif canvas_type == "pyplot":
        #    canvas.set_title(title)

        # transform to local ENU coordinates at every point, and start at 0.
        pos = self.data[['pos_x', 'pos_y', 'pos_z']].values
        if pos.shape[0] == 0:
            return False

        if canvas_type == "pyplot":
            canvas.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                        color=color)
        else:
            raise NotImplementedError

        pass

    def slide(self, window=60 * 10 ** 9,
              time_step=1 * 10 ** 9,
              time_start=None,
              time_stop=None,
              overlap_ends=True):
        r""" Produce equal-time SpaceCurves using a sliding-window on self.
        
        Parameters
        ----------
        window : int
            length of window in nanoseconds (default = 60*10**9 or 1 minute)
        time_step : int
            step size in nanoseconds between window starting-points (default = 1*10** or 1 second)
        time_start : int
            start time in nanoseconds (default = None, min time)
        time_stop : int
            stop time in nanoseconds (default = None, max time)
        overlap_ends : bool
            Should endpoints overlap (True) or abutt (False) (default = True)

        Returns
        -------
        iterator of SpaceCurves
        To access the tracks as a list, do
        :code:`list( FT.snip() )`

        Notes
        -----
        The trackids are unchanged! You might want to change them manually.
        """

        if time_start is None:
            time_start = self.data.index.values[0]
        if time_stop is None:
            time_stop = self.data.index.values[-1]

        if not overlap_ends:
            window -= 1

        slices = np.arange(time_start, time_stop, time_step)
        for i, start_ns in enumerate(slices):
            stop_ns = start_ns + window
            newFT = self.copy()
            newFT.data = self.data.loc[start_ns:stop_ns].copy()
            newFT.info = self.info.loc[start_ns:stop_ns].copy()
            yield newFT

            if stop_ns >= time_stop:
                break

    def snip(self, time_step=60 * 10 ** 9, time_start=None, time_stop=None,
             overlap_ends=True):
        """ Cut this SpaceCurve into equal-time snippets.
        This is just a simple wrapper around self.slide( )
        with time_step == window.

        time_step=60*10**9 == 1 minute

        Note! The trackids are unchanged!
        You might want to change them manually.

        Yields
        ------
        iterator of SpaceCurves
        To access the tracks as a list, do
        list( FT.snip() )
        """

        for F in self.slide(window=time_step,
                            time_step=time_step,
                            time_start=time_start,
                            time_stop=time_stop,
                            overlap_ends=overlap_ends):
            yield F

    def clean_copy(self, cleanup_func=curve_geometry.mollifier, **kwargs):
        r""" Make copy in which a cleanup function performed
        on the data.

        cleanup_func is the interface to the functions in curve_geometry.
        In particular:
        1. :func:`curve_geometry.cleanup` does slope comparison
        2. :func:`curve_geometry.mollifier` does a convolution

        cleanup_func should at the minimum take time and position as
        positional arguments 1 and 2 and return both new time and new
        position arrays.  All Keyword arguments (:code:`**kwargs`) are passed on to
        cleanup_func.

        One can tune the various cleanup functions by passing
        a user manufactured function, for example
        .. code::
            
            my_cleanup = lambda time, x:curve_geometry.mollifier(time,x,width=0.5)
            clean_track = track.clean_copy(cleanup_func = my_cleanup)
        
        would be equivalent to
        .. code::
        
            clean_track = track.clean_copy(cleanup_func=curve_geometry.mollifier, width=0.5)

        """

        # Compute the cleaned up values
        time, clean_px = cleanup_func(self.data['time'].values,
                                      self.data['pos_x'].values, **kwargs)
        timey, clean_py = cleanup_func(self.data['time'].values,
                                       self.data['pos_y'].values, **kwargs)
        timez, clean_pz = cleanup_func(self.data['time'].values,
                                       self.data['pos_z'].values, **kwargs)
        assert np.all(time == timey)
        assert np.all(time == timez)

        # [time, clean_vx] = cleanup_func(self.data['time'].values,
        #                                self.data['vel_x'].values)
        # [time, clean_vy] = cleanup_func(self.data['time'].values,
        #                                self.data['vel_y'].values)
        # [time, clean_vz] = cleanup_func(self.data['time'].values,
        #                                self.data['vel_z'].values)

        # Build a new index from the time array
        new_index_start = self.data.index.values[0]
        new_index = np.int64(time*10**9)+new_index_start

        assert new_index.shape[0] == time.shape[0],\
            """New index is shape {} but time is shape {}""".format(new_index.shape[0], time.shape[0])


        # Instantiate a new flighttrack object from the cleaned versions
        newFT = self.__class__(new_index,
                               trackid=self.trackid,
                               platform=self.platform,
                               activity=self.activity,
                               mollified=True)
        newFT.data = pd.DataFrame({'time': time,
                                   'pos_x': clean_px,
                                   'pos_y': clean_py,
                                   'pos_z': clean_pz,
                                   'quality': -np.ones(time.shape, dtype=np.int64)},
                                  columns=['time', 'pos_x', 'pos_y', 'pos_z', 'quality'],
                                  index=new_index)
        return newFT

    def compute(self):
        """ Compute some nice invariants and store them to self.info. 
        Vector quantities are generally in the fixed frame.
        """
        # raw data in ECEF frame.
        T = self.data['time'].values
        P = self.data[['pos_x', 'pos_y', 'pos_z']].values
        V = curve_geometry.secant_derivative(T, P)
        #V2 = self.data[['vel_x', 'vel_y', 'vel_z']].values
        #assert not np.any(np.isnan(V2)), "{} bad in {}".format(np.where(np.isnan(V2)), V2.shape)

        # all derivatives and integrals
        A = curve_geometry.secant_derivative(T, V)
        J = curve_geometry.secant_derivative(T, A)
        arclengthS = curve_geometry.secant_arclength(P)

        # norms
        #recspeed = np.linalg.norm(V2, axis=1).flatten()
        speed = np.linalg.norm(V, axis=1).flatten()
        acc = np.linalg.norm(A, axis=1).flatten()
        jerk = np.linalg.norm(J, axis=1).flatten()


        if len(self) > 4:
        # Use signature curve to make curv and tors
            kap, kap_s, tau, tau_s = self.signature_curve()
       
        #KT = np.ndarray(shape=(T.shape[0], 2), dtype='float')
        #KT[:2, :] = sc[0, :]
        #KT[:-2, :] = sc[-1, :]
        #KT[2:-2, :] = sc


        #TNB_enu = curve_geometry.frenet_frame(V, A)
        #tilt = np.arccos(np.abs(TNB_enu[:, 2, 2]))

        #dKT_ds = curve_geometry.secant_derivative(arclengthS, KT)
        #tilt_array = tilt.copy()
        #tilt_array.shape = (tilt.shape[0], 1)
        #dtilt_ds = curve_geometry.secant_derivative(arclengthS,
        #tilt_array).flatten()

#        curv_per_alt = KT[:, 0] / (P[:,2]+1)
#        acc_per_alt = acc / (P[:,2]+1)
#        tors_per_alt = KT[:, 1] / (P[:,2]+1)
        #print(tau.shape)
        #print(kap_s.shape)
        #print(tau.shape)
        #print(tau_s.shape)
#        friction = kap * speed ** 2  ## need to check this for angle.
#        bank = np.arctan(friction / 9.8)  ## replace 9.8 with gravity??

        
        # dKT_ds = curve_geometry.secant_derivative(arclengthS, KT)
        
        self.info['vel_x'] = V[:, 0]
        self.info['vel_y'] = V[:, 1]
        self.info['vel_z'] = V[:, 2]
        self.info['acc_x'] = A[:, 0]
        self.info['acc_y'] = A[:, 1]
        self.info['acc_z'] = A[:, 2]
        self.info['len'] = arclengthS  # This is the only thing we store in data
        self.info['speed'] = speed
        #self.info['recspeed'] = recspeed
        self.info['acc'] = acc
        self.info['jerk'] = jerk

        if len(self) > 4:
            self.info['curv'] = kap
            self.info['curv_s'] = kap_s
            self.info['tors'] = tau
        
            self.info['tors_s'] = 0.0
        #self.info['tors_s'].values[3:-3] = tau_s
        #self.info['tors_s'].values[:3] = tau_s[0]
        #self.info['tors_s'].values[-3:] = tau_s[-1]


        #self.info['tors'] = tau
        #self.info['dKds'] = curv_s
        #self.info['dTds'] = tau_s
        pass

    def featurize(self, sort_and_grab_num=None):
        """ A convenience function to compute everything we think might be
        important.  This returns a dictionary of np arrays. 
        Optionally, we can hit all of the persistence diagrams with sort+grab.
        """

        feature_dict = dict()

        self.compute()

        for name in ['curv', 'tors', 'up', 'acc',
                     'Dcurv/Dlen', 'Dtors/Dlen', 'Dup/Dlen',
                     'speed', 'friction', 'curv/alt', 'tors/alt', 'acc/alt',
                     ]:
            S = Signal(self.info[name].values)
            if sort_and_grab_num is not None:
                feature_dict[name] = S.pers.grab(sort_and_grab_num)['keepcode']
            else:
                feature_dict[name] = S.pers.diagram

        return feature_dict

    def copy(self):
        """ make an identical copy of self. """
        d = self.data.copy()
        newFT = self.__class__(d.index.values, trackid=self.trackid,
                               platform=self.platform,
                               activity=self.activity,
                               )
        newFT.data = d
        newFT.info = self.info.copy()
        return newFT

    @classmethod
    def load(cls, filename):
        """ Simple CSV reader.
        columns ('time', 'pos_x', 'pos_y', 'pos_z')
        """
        data = pd.read_csv(filename, sep="\t")
        F = cls(tn=data['time'].values,
                px=data['pos_x'].values,
                py=data['pos_y'].values,
                pz=data['pos_z'].values)
        return F

    def duration(self):
        """ return string of duraction of SpaceCurve,
        converting integer nanoseconds to string seconds.
        """
        times = self.data.index.values
        if len(times) == 0:
            return 0
        ns = repr(times[-1] - times[0])
        assert ns.find("e") == -1
        seconds = ns[:-9] + "." + ns[-9:]
        return seconds

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "SpaceCurve with {} entries and duration {}".format(len(self), self.duration())

    def mass_profile(self, num_bins=100, underflow=-50, overflow=50):
        """ histogram of the lift_and_mass by mass. """

        LM = self.lift_and_mass()
        width = np.float64(overflow - underflow) / np.float64(num_bins)

        finite_bins = np.arange(underflow, overflow + width, width)
        bins = np.zeros(shape=(finite_bins.shape[0] + 2,))
        bins[0] = np.float64('-inf')
        bins[-1] = np.float64('inf')
        bins[1:-1] = finite_bins[:]

        return np.histogram(LM[:, 1], bins)

    def lift_profile(self, num_bins=500, underflow=0, overflow=500000):
        """ histogram of the lift_and_mass by lift. """
        LM = self.lift_and_mass()
        width = np.float64(overflow - underflow) / np.float64(num_bins)

        finite_bins = np.arange(underflow, overflow + width, width)
        bins = np.zeros(shape=(finite_bins.shape[0] + 2,))
        bins[0] = np.float64('-inf')
        bins[-1] = np.float64('inf')
        bins[1:-1] = finite_bins[:]

        return np.histogram(LM[:, 0], bins)

    def lift_and_mass(self):
        r""" Produce the terms of the force-balance equations, to help derive
        coefficient-of-lift and mass from pure trajectory information.

        Consider an aircraft flying with velocity :math:`v`, banking angle
        :math:`\theta`, air density :math:`\rho`, mass :math:`m`, gravity
        :math:`g`, vertical acceleration :math:`\ddot{z}`, and lift coefficient
        :math:`C_{\text{lift}}`.

        The force-balance equation is 
        ..math:: 

            LHS = \frac12 \|v\|^2 \, \cos(\theta)\, \rho
            RHS = (g + \ddot{z})
            LHS\, C_{\text{lift}} = RHS\, m
      
        Returns
        -------
        coefficients : :class:np.array
            the columns are LHS and RHS. 

        """

        # build an atmospheric air density profile using the data
        # 1976 international standard atmosphere.  We just use a 4th-order fit.
        # https://en.wikipedia.org/wiki/International_Standard_Atmosphere
        # alts = np.array([-610, 11000, 20000, 32000,47000])
        # dens = np.array([1.2985,0.3639,0.0880,.0105,0.0020])
        # air_denstiy = np.poly1d(np.polyfit(alts,dens,4))
        air_density = np.poly1d([2.70588959e-19, -5.57103078e-14, 3.91598431e-09, -1.15140013e-04, 1.22679477e+00])

        speed = self.info['speed'].values
        altacc = self.info['acc_z'].values
        bank = self.info['bank'].values
        cosb = np.cos(bank)
        h = self.info['alt'].values
        air = air_density(h)
        LHS = 0.5 * speed ** 2 * cosb * air
        RHS = 9.8 + altacc
        return np.array([LHS, RHS]).T

    def auto_bin(self, num_bins=10, underflow=0, overflow=100):
        """ Count bins on the transformed persistence diagrams of 
        -Speed/7         (so it is expected to be between 0 and 100)
        -Climb/3         (so it is expected to be between 0 and 100)
        -Curvature*10000 (so it is expected to be between 0 and 100)
        -Torsion*10000   (so it is expected to be between 0 and 100)
        -Bank*100/(pi/4) (==grade, between 0 and 100)

        Bins are added for (-inf, underflow] and (overflow,inf).
        If underflow and overflow are not provided, they are set to min/max.

        """

        speed01 = Signal(self.info['speed'].values / 7)
        climb01 = Signal(self.info['climb'].values / 3)
        curv01 = Signal(self.info['curv'].values * 10000)
        tors01 = Signal(self.info['tors'].values * 10000)
        bank01 = Signal(self.info['bank'].values * 400 / np.pi)

        width = np.float64(overflow - underflow) / np.float64(num_bins)

        speed_hist, speed_bins = speed01.pers.bin(width=width,
                                                  underflow=underflow,
                                                  overflow=overflow)
        climb_hist, climb_bins = climb01.pers.bin(width=width,
                                                  underflow=underflow,
                                                  overflow=overflow)
        curv_hist, curv_bins = curv01.pers.bin(width=width,
                                               underflow=underflow,
                                               overflow=overflow)
        tors_hist, tors_bins = tors01.pers.bin(width=width,
                                               underflow=underflow,
                                               overflow=overflow)
        bank_hist, bank_bins = bank01.pers.bin(width=width,
                                               underflow=underflow,
                                               overflow=overflow)
        assert np.all(curv_bins == tors_bins) and np.all(
            curv_bins == bank_bins)

        return {'bins': curv_bins,
                'speed': speed_hist,
                'climb': climb_hist,
                'curv': curv_hist,
                'tors': tors_hist,
                'bank': bank_hist}

    def signature_curve(self):
        r""" Olver/Boutin signature curve.
        (kappa, kappa_s, tau, tau_s)
        due to difference methods, the lengths are (n-2, n-4, n-4, n-6),
        but we pad them all to length n

        Usage:
        ------
        >>> ts = np.arange(0,12,0.1)
        >>> # Line
        >>> C = SpaceCurve(tn = np.arange(ts.shape[0]),
        ...                px = 5*ts + 3, py=2*ts + 5)
        >>> kappa, kappa_s, tau, tau_s = C.signature_curve()
        >>> np.allclose(kappa, 0.0, atol=1.5e-7)
        True
        >>> np.allclose(kappa_s, 0.0, atol=1.5e-7)
        True
        >>> np.allclose(tau, 0.0, atol=1.5e-7)
        True
        >>> np.allclose(tau_s, 0.0, atol=1.5e-7)
        True
        >>> ts = np.arange(0,12,0.1)
        >>> # Circle with constant speed
        >>> C = SpaceCurve(tn = np.arange(ts.shape[0]),
        ...                px = np.cos(ts),
        ...                py = np.sin(ts))
        >>> kappa, kappa_s, tau, tau_s = C.signature_curve()
        >>> kappa.shape # padded
        (120,)
        >>> kappa_s.shape
        (120,)
        >>> np.allclose(kappa, 1.)
        True
        >>> np.allclose(kappa_s, 0.)
        True
        >>> ts = np.arange(0,12,0.1)
        >>> # Circle with varying speed
        >>> C = SpaceCurve(tn = np.arange(ts.shape[0]),
        ...                px = np.cos(ts**2),
        ...                py = np.sin(ts**2))
        >>> kappa, kappa_s, tau, tau_s  = C.signature_curve()
        >>> kappa_s.shape # padded
        (120,)
        >>> np.allclose(kappa, 1.)
        True
        >>> np.allclose(kappa_s, 0.)
        True
        >>> np.allclose(tau, 0.)
        True
        >>> np.allclose(tau_s, 0.)
        True
        >>> ts = np.arange(1,13,0.01)
        >>> # A Spiral
        >>> C = SpaceCurve(tn = np.arange(ts.shape[0]),
        ...                px = np.exp(0.75*ts)*np.cos(ts),
        ...                py = np.exp(0.75*ts)*np.sin(ts))
        >>> kappa, kappa_s, tau, tau_s = C.signature_curve()
        >>> kappa.shape # padded
        (1200,)
        >>> np.allclose(kappa[1:-1],  np.exp(-0.75*ts[1:-1]), atol=0.1)
        True
        >>> np.allclose(kappa_s[2:-2]*np.exp(1.5*ts[2:-2]), -12./25., atol=0.01)
        True
        >>> # A Helix
        >>> C = SpaceCurve(tn = np.arange(ts.shape[0]),
        ...                px = 3*np.cos(ts),
        ...                py = 3*np.sin(ts),
        ...                pz = 4*ts)
        >>> kappa, kappa_s, tau, tau_s = C.signature_curve()
        >>> np.allclose(kappa, 3/25.)
        True
        >>> np.allclose(kappa_s, 0.0)
        True
        >>> np.allclose(tau, 4/25.)
        True
        >>> np.allclose(tau_s, 0.0)
        True
        >>> # A Helix (reversed)
        >>> C = SpaceCurve(tn = np.arange(ts.shape[0]),
        ...                px = 3*np.cos(-ts),
        ...                py = 3*np.sin(-ts),
        ...                pz = 4*ts)
        >>> kappa, kappa_s, tau, tau_s = C.signature_curve()
        >>> np.allclose(kappa, -3/25.,)
        True
        >>> np.allclose(kappa_s, 0.0)
        True
        >>> np.allclose(tau, -4/25.,)
        True
        >>> np.allclose(tau_s, 0.0)
        True
        """
        
        #if not np.all(self.data['pos_z'].values == 0.):
        #    raise ValueError("This method currently handles only planar curves.")

        # if np.all(self.data['pos_x'].values == 0.):
        pos = self.data[['pos_x', 'pos_y', 'pos_z']].values
        n = pos.shape[0]

        # follow Calabi's naming convention.
        # We deal with 1-interior points.
        P_i_mns_1 = pos[:-2, :]
        P_i       = pos[1:-1, :]
        P_i_pls_1 = pos[2:, :]

        # Use the determinant to set a right-handed sign.
        triples = np.ndarray(shape=(n-2, 3,3), dtype='float')
        triples[:, 0, :] = P_i_mns_1
        triples[:, 1, :] = P_i      
        triples[:, 2, :] = P_i_pls_1
        sign = (-1)**np.signbit(np.linalg.det(triples))


        a = np.sqrt(np.sum((P_i       - P_i_mns_1)**2, axis=1))
        b = np.sqrt(np.sum((P_i_pls_1 - P_i      )**2, axis=1))
        c = np.sqrt(np.sum((P_i_pls_1 - P_i_mns_1)**2, axis=1))
        s = 0.5*(a+b+c)
        # If a,b,c are co-linear, then we might get s-c to be negative
        # due to roundoff error.  (or other permutations)
        s_minus_a = np.clip(s-a, 0., np.infty)
        s_minus_b = np.clip(s-b, 0., np.infty)
        s_minus_c = np.clip(s-c, 0., np.infty)

        abc = a*b*c
        # Calabi,et al eqn (2.2)
        non_trivial = (abc != 0)
        kappa = sign*4*np.sqrt(s*s_minus_a*s_minus_b*s_minus_c)
        kappa[non_trivial] = kappa[non_trivial] / abc[non_trivial]
        kappa[~ non_trivial] = 0.0
        assert kappa.shape[0] == n-2

        # Now, we follow Boutin's naming convention.
        # We deal with 2-interior points.
        P_i = pos[2:-2, :]
        P_i_pls_1 = pos[3:-1, :]
        P_i_pls_2 = pos[4:, :]
        P_i_mns_1 = pos[1:-3, :]
        P_i_mns_2 = pos[0:-4, :]
        a = np.sqrt(np.sum((P_i       - P_i_mns_1)**2, axis=1))
        b = np.sqrt(np.sum((P_i_pls_1 - P_i      )**2, axis=1))
        # c = np.sqrt(np.sum((P_i_pls_1 - P_i_mns_1)**2, axis=1))
        d = np.sqrt(np.sum((P_i_pls_2 - P_i_pls_1)**2, axis=1))
        e = np.sqrt(np.sum((P_i_pls_2 - P_i      )**2, axis=1))
        f = np.sqrt(np.sum((P_i_pls_2 - P_i_mns_1)**2, axis=1))
        g = np.sqrt(np.sum((P_i_mns_2 - P_i_mns_1)**2, axis=1))
        # reverse collections, for reverse tau
        dd = g
        ee = np.sqrt(np.sum((P_i_mns_2 - P_i      )**2, axis=1))
        ff = np.sqrt(np.sum((P_i_mns_2 - P_i_pls_1)**2, axis=1))

        assert a.shape[0] == n-4

        # Note that the index of a goes 0..n-5, and
        # Note that the index of kappa goes 0..n-3.
        # and P[i] corresponds to a[i] and kappa[i+1]
        denom_ks = 2*a + 2*b + d + g
        non_trivial = (denom_ks != 0)
        kappa_s = 3*(kappa[2:] - kappa[:-2])
        kappa_s[non_trivial] = kappa_s[non_trivial]/denom_ks[non_trivial]
        kappa_s[~ non_trivial] = 0.0
        
      
        # tau according to Boutin's \tilde{tau}_1, in the forward direction
        tetra_height = np.ndarray(shape = kappa_s.shape, dtype='float')
        for i in range(P_i.shape[0]):
            tetrahedron = np.array([P_i_mns_1[i] - P_i[i], 
                                    P_i_pls_1[i] - P_i[i], 
                                    P_i_pls_2[i] - P_i[i]]).T
            
            tetra_height[i] = np.linalg.qr(tetrahedron, mode='r')[-1,-1]
        # we want tau = 6 * tetra_height / denom_t, but 
        # don't want to divide by zero, which happens if points repeat.
        tau_fwd = 6 * tetra_height
        denom_t = d * e * f * kappa[1:-1] # sign is inherited!
        non_trivial = (denom_t != 0) & (tetra_height != 0)
        tau_fwd[non_trivial] = tau_fwd[non_trivial] / denom_t[non_trivial]
        tau_fwd[~ non_trivial] = 0.0 
        
        # tau according to Boutin's \tilde{tau}_1, in the backard direction
        tetra_height = np.ndarray(shape = kappa_s.shape, dtype='float')
        for i in range(P_i.shape[0]):
            tetrahedron = np.array([P_i_mns_2[i] - P_i[i], 
                                    P_i_mns_1[i] - P_i[i], 
                                    P_i_pls_1[i] - P_i[i]]).T
            
            tetra_height[i] = np.linalg.qr(tetrahedron, mode='r')[-1,-1]
        # we want tau = 6 * tetra_height / denom_t, but 
        # don't want to divide by zero, which happens if points repeat.
        tau_bwd = 6 * tetra_height
        denom_t = d * e * f * kappa[1:-1] # sign is inherited!
        non_trivial = (denom_t != 0) & (tetra_height != 0)
        tau_bwd[non_trivial] = tau_bwd[non_trivial] / denom_t[non_trivial]
        tau_bwd[~ non_trivial] = 0.0 
        
        tau = (tau_fwd + tau_bwd)/2

       
        
#        # tau_s according to Boutin's (17), in the forward direction
#        P_i_pls_3 = pos[4:, :]
#        h = np.sqrt(np.sum((P_i_pls_3 - P_i_pls_2[:-1])**2, axis=1))
#        dh = d[:-1] + h
#        denom_ts = denom_ks[1:-1] + dh
#        old_settings = np.seterr(divide='ignore')  #seterr to known value
#        tau_s_fwd = 4*(tau[1:] - tau[:-1] + (denom_ks[1:-1] -3*dh) * tau * kappa_s / (6 * kappa[2:-2]))/denom_ts
#        tau_s_fwd[denom_ts == 0] = 0
#        np.seterr(**old_settings)
#
#        # tau_s according to Boutin's (17), in the backward direction
#        P_i_mns_3 = pos[:-4, :]
#        h = np.sqrt(np.sum((P_i_mns_3 - P_i_mns_2[1:])**2, axis=1))
#        dh = d[1:] + h
#        denom_ts = denom_ks[1:-1] + dh
#        old_settings = np.seterr(divide='ignore')  #seterr to known value
#        tau_s_bwd = 4*(tau[1:] - tau[:-1] + (denom_ks[1:-1] -3*dh) * tau * kappa_s / (6 * kappa[2:-2]))/denom_ts
#        tau_s_bwd[denom_ts == 0] = 0
#        np.seterr(**old_settings)
#        
#        tau_s = (tau_s_fwd + tau_s_bwd)/2

        assert kappa.shape == (n-2,)
        assert kappa_s.shape == (n-4,)
        assert tau.shape == (n-4,)
        #assert tau_s.shape == (n-6,)

        #print(kappa.shape, tau_s.shape)

        kappa_pad = np.ndarray(shape=(n,), dtype='float')
        kappa_s_pad = np.ndarray(shape=(n,), dtype='float')
        tau_pad = np.ndarray(shape=(n,), dtype='float')
        tau_s_pad = np.ndarray(shape=(n,), dtype='float')
        
        kappa_pad[1:-1] = kappa
        kappa_pad[:1] = kappa[0]
        kappa_pad[-1:] = kappa[-1]

        kappa_s_pad[2:-2] = kappa_s
        kappa_s_pad[:2] = kappa_s[0]
        kappa_s_pad[-2:] = kappa_s[-1]

        tau_pad[2:-2] = tau
        tau_pad[:2] = tau[0]
        tau_pad[-2:] = tau[-1]

        tau_s_pad[:] = 0.0
        #tau_s_pad[2:-2] = tau_s
        #tau_s_pad[:2] = tau_s[0]
        #tau_s_pad[-2:] = tau_s[-1]
        
        return kappa_pad, kappa_s_pad, tau_pad, tau_s_pad

# end of class SpaceCurve
