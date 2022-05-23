r""" 
This module defines the the basic objects for persistence diagrams:
    
    - :class:`PersDiag`

Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE
"""

#__import__('pkg_resources').declare_namespace(__name__)
#from .persdiag import PersDiag


import numpy as np
import pandas as pd

from .dim0 import mkforestDBL

class PersDiag(object):
    """ Persistence Diagrams and related merge-tree information.  Sometimes
    known as barcodes.
    
    A :class:`PersDiag` consists of a :class:`pandas.DataFrame` of birth and
    death values (sorted by length), along with the tree and forest for range and domain inclusion.  
    
    You should not call this directly --- instead use the constructor on your
    `homology.SimplicialComplex` or 'timeseries.Signal` object.
    
    The input parameters are of the form that comes from 
    `persistence.dim0.unionfind` or similar
    
    Notes
    ------
    Users should never call this directly --- instead use the constructor on your
    :class:`homology.SimplicialComplex` or :class:`timeseries.Signal` object.
 

    Parameters
    ----------
    birth_index : list_like
    death_index : list_like
    birth_value : list_like
    death_value : list_like
    mergetree : dict
    

    Attributes
    ----------
    diagram : :class:`pandas.DataFrame`
        A DataFrame showing all of the birth/death information.
        It has columns :code:`birth_index`, :code:`death_index`, :code:`birth`,
        :code:`death`, and :code:`pers`

    lefts : :class:`pandas.Series`
    rights : :class:`pandas.Series`
    domains : dict
    bartree : dict
    bartree_parents : dict
    forest : dict
    forest_parents : dict
    mergetree : dict


    See Also
    --------
    :func:`homology.dim0.unionfind`

    """

    def __init__(self, birth_index, death_index, birth_value, death_value, mergetree):

        self.diagram = pd.DataFrame(
            {'birth_index': birth_index, 'death_index': death_index, 'birth': birth_value, 'death': death_value},
            columns=['birth_index', 'death_index', 'birth', 'death'])

        self.diagram['pers'] = self.diagram['death'] - self.diagram['birth']

        self.diagram = self.diagram.sort_values(by='pers')
        
        self.lefts = self.diagram[['birth_index', 'death_index']].min(axis=1)
        self.rights = self.diagram[['birth_index', 'death_index']].max(axis=1)
        
        self.domains = dict([((self.lefts[i], self.rights[i]), i) for i in self.lefts.index])
         
        self.bartree, self.bartree_parents = mkforestDBL(
            self.diagram.index.values.astype(np.int64), 
            self.diagram['birth'].values.astype(np.float64), 
            self.diagram['death'].values.astype(np.float64))

        self.forest, self.forest_parents = mkforestDBL(
            self.lefts.index.values.astype(np.int64),
            self.lefts.values.astype(np.float64),
            self.rights.values.astype(np.float64))

        self.mergetree = mergetree

        self._clip = dict()

    def grab(self, n):
        """ clip the barcode, using only the top n entries. 
        this is "sort-and-grab'.
        """
        if n >= len(self.diagram):
            beta = 0
        else:
            beta = self.diagram['pers'].values[-n-1:-n].mean()
        return self.clip(beta)
    
    def clip(self, beta):
        """ clip the barcode below beta. """

        # simple cache
        if beta in self._clip:
            return self._clip[beta]

        bc = self.diagram
        killbc = bc[bc['pers'] <= beta]  # boolean selector

        keepbc = bc.loc[~bc.index.isin(killbc.index)]

        # trim the tree to find the roots of those we will kill. 
        # Not sure if this is really faster?
        ignore = pd.Series(index=killbc.index, dtype='bool')
        ignore[::] = False
        tree = self.forest
        for idx in killbc.index[::-1]:  # from long to short.
            for child_idx in tree[idx]:
                ignore.loc[child_idx] = True
        ignorebc = killbc.loc[ignore]
        killbc = killbc.loc[~ignore]

        self._clip[beta] = {'killcode': killbc,
                            'keepcode': keepbc,
                            'ignorecode': ignorebc,
                            'beta': beta}
        return self._clip[beta]

    def bin_len(self, width, underflow=None, overflow=None):
        bin_count, bins = self.bin(width,
                                   underflow=underflow,
                                   overflow=overflow,
                                   transform="project")
        return bin_count.sum(axis=0), bins

    def bin_bot(self, width, underflow=None, overflow=None):
        bin_count, bins = self.bin(width,
                                   underflow=underflow,
                                   overflow=overflow,
                                   transform="project")

        return bin_count.sum(axis=1), bins

    def bin_top(self, width, underflow=None, overflow=None):
        bin_count, bins = self.bin(width,
                                   underflow=underflow,
                                   overflow=overflow,
                                   transform=None)

        return bin_count.sum(axis=0), bins

    def bin_diag(self, width, underflow=None, overflow=None):
        bin_count, bins = self.bin(width,
                                   underflow=underflow,
                                   overflow=overflow,
                                   transform="rotate")
        return bin_count.sum(axis=1), bins
        pass

    def transform(self, transform=None, points=None):
        r"""
        Produce an array of points, obtained by transforming the persistence
        diagram.

        

        If transform==None [default], then we leave the diagram alone.
        (birth, death)
        If transform=="project", then we project the diagram so that
        (birth, death) --> (birth, death-birth) = (birth, pers)
        If transform=="rotate", then we rotate-and-rescale so that
        (birth, death) --> ( (death+birth)/2, (death-birth)/2 ) 
        If transform is a 2x2 array, apply it.

        If provided with an array "points", transform that instead of the
        diagram.

        """

        if points is None:
            points = self.diagram[['birth', 'death']].values

        if transform is None:
            rectified = points
        elif transform == "rotate":
            lintrans = np.array([[0.5, 0.5], [-0.5, 0.5]])
            rectified = (lintrans.dot(points.T)).T
        elif transform == "project":
            lintrans = np.array([[1, 0], [-1, 1]])
            rectified = (lintrans.dot(points.T)).T
        elif type(transform) == np.ndarray and len(transform.shape) == 2:
            lintrans = transform
            rectified = (lintrans.dot(points.T)).T
        else:
            raise TypeError("transform must be a string or a 2x2 array.")
        return rectified

    def bin(self, width, underflow=None, overflow=None, transform="project"):
        r""" Count bins on the transformed persistence diagram.
        Bins are added for (-inf, underflow] and (overflow,inf).
        If underflow and overflow are not provided, they are set to min/max.

        NOTE!  The bin_count numbers look "upside-down and backwards" versus
        the projected persistence diagram.  This is because indexing of a
        matrix is from the top, whereas we read pictures from the bottom. 

        return:
            bin_count (2-dim np array of bin counts),
            bin_boundaries
        """
        if underflow is None:
            underflow = self.diagram['birth'].values.min()
        if overflow is None:
            overflow = self.diagram['death'].values.max()
        
        finite_bins = np.arange(underflow, overflow+width, width)
        bins = np.zeros(shape=(finite_bins.shape[0]+2,))
        bins[0] = np.float64('-inf')
        bins[-1] = np.float64('inf')
        bins[1:-1] = finite_bins[:]

        rectified = self.transform(transform)

        bin_count, x_bins, y_bins = np.histogram2d(rectified[:, 0],
                                                   rectified[:, 1],
                                                   (bins, bins))
        assert all(x_bins == bins) and all(y_bins == bins)
        return bin_count, bins

    def syzygy(self, powers):
        """ compute the syzygy coordinate of my barcode. 
        len*min**(powers[0])*max**powers[1]
        """
        bc = self.diagram
        return bc['pers']*bc['birth']**powers[0]*bc['death']**powers[1]

    def plot(self, canvas, transform=None, bins=None, title="Pers Diag"):
        r""" Plot the persistence diagram using `matplotlib` or `bokeh`.
        
        Parameters
        ----------
        canvas      An instance of :class:`bokeh.plotting.figure.Figure`
                    invoked as `canvas = bokeh.plotting.figure()`, or a 
                    an instance of :class:`matplotlib.axes._subplots.AxesSubplot`
                    invoked as `fig,canvas = matplotlib.pyplot.subplots()` or
                    similar.

        
        Notes
        -----
        You have to save or show the canvas after running this call.
        """

        if type(canvas).__module__ == 'bokeh.plotting.figure':
            canvas_type = "bokeh"
            import bokeh.plotting
            from bokeh.models import Span
        elif type(canvas).__module__ == 'matplotlib.axes._subplots':
            canvas_type = "pyplot"
            import matplotlib.pyplot as plt
        else:
            raise NotImplementedError("""canvas must be a bokeh.plotting.figure() or a matplotlib.pyplot.subplots()[1].
        You gave me {}""".format(type(canvas))
                ) 

        if canvas_type == "bokeh":
            canvas.title = title
        elif canvas_type == "pyplot":
            canvas.set_title(title)

        if bins is not None:
            if canvas_type == "bokeh":
                for edge in bins:
                    if np.isfinite(edge):
                        canvas.renderers.append(Span(location=edge,
                                                     dimension='height',
                                                     line_color='purple',
                                                     line_alpha=0.8))
                        canvas.renderers.append(Span(location=edge,
                                                     dimension='width',
                                                     line_color='purple',
                                                     line_alpha=0.8))
            elif canvas_type == "pyplot":
                for edge in bins:
                    if np.isfinite(edge):
                        canvas.axhline(edge, color='purple', alpha=0.8)
                        canvas.axvline(edge, color='purple', alpha=0.8)

        absmin = self.diagram['birth'].min()
        absmax = self.diagram['death'].max()
        boundary = np.array([[absmin, absmin],
                             [absmin, absmax],
                             [absmax, absmax]])
        boundary = self.transform(transform, boundary)
        if canvas_type == "bokeh":
            canvas.line(boundary[:, 0], boundary[:, 1], color='orange')
        elif canvas_type == "pyplot":
            canvas.plot(boundary[:, 0], boundary[:, 1], color='orange')

        diagonal = np.array([[absmin, absmin],
                             [absmax, absmax]])
        diagonal = self.transform(transform, diagonal)
        if canvas_type == "bokeh":
            canvas.line(diagonal[:, 0], diagonal[:, 1], color='red')
        elif canvas_type == "pyplot":
            canvas.plot(diagonal[:, 0], diagonal[:, 1], color='red')
         
        rectified = self.transform(transform)

        if canvas_type == "bokeh":
            canvas.circle(rectified[:, 0], rectified[:, 1], color='blue',
                          alpha=0.5, size=3)
        elif canvas_type == "pyplot":
            canvas.scatter(rectified[:, 0], rectified[:, 1], color='blue', alpha=0.5)
        pass
