r"""This module contains classifiers based on the SciKitLearn framework.

    - :class:`CDER`
    - :class:`GaussianMixtureClassifier`

We use these for modelling marked-point processes and labelled pointclouds.

Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE
"""

from __future__ import print_function

import logging
import numpy as np
import sklearn
import multidim
import multidim.covertree 
from .fast_algorithms import gaussian, gaussian_fit, entropy, distance_cache_None


class GaussianMixtureClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    r""" A scikit-learn classification estimator, for any sort of Gaussian
    mixture model.

    A Gaussian mixture module is a collection of labelled Gaussian functions 

    This is provided mainly to provide a parent class for `CDER`.

    """ 

    def __init__(self, stop_level=None, **kwargs):
        self.stop_level = stop_level
        self.kwargs = kwargs
        self.pointcloud = None
        self.all_labels = None
        self.covertree = None
        self.gaussians = None

    def get_params(self, deep=False):
        r""" Pass original kwargs internally."""
        return self.kwargs

    def gausscoords(self):
        """ This is the method to overwrite for specific models! """
        gaussians = [{'mean': None,  # mean point as numpy 1d
                      'std': None,   # signal diagonals as numpy 1d
                      'rotation': None,  # rotation as numpy 2d
                      'weight': None,  # weight as calculated by fit
                      'label': None,  # best label from labels_train
                      'adult': None,  # adult coordinates
                      'index': None,  # adult index
                      'radius': None,  # radius of region
                      }]
        tree = None  # some sort of object for re-use.  It should hava a pointcloud attribute of PointCloud type.
        return tree, gaussians

    def fit(self, *training):
        """ Fit this estimator to training data.  This is the step that runs
        the actual predictor.
        
        Parameters
        ----------
        *training 
           Can be any of the following: 

            - a pair of list-like objects: pointclouds_train and labels_train
              The items in pointclouds_train should be :class:`numpy.ndarray`
              objects.  The items in labels_train should be strings
              (preferably, color names).
            - a single labelled and weighted :class:`multidim.PointCloud`
            - a single :class:`multidim.covertree.CoverTree` obtained from a
              labelled and weighted :class:`multidim.PointCloud`

        """
        if len(training) == 1 and type(training[0]) == multidim.covertree.CoverTree:
            self.covertree = training[0]
            self.pointcloud = self.covertree.pointcloud
            self.all_labels = self.pointcloud.label_info.index.values
        elif len(training) == 1 and type(training[0]) == multidim.PointCloud:
            self.pointcloud = training[0]
            self.all_labels = self.pointcloud.label_info.index.values
            self.covertree = multidim.covertree.CoverTree(self.pointcloud)
        elif len(training) == 2:
            pointclouds_train = training[0]
            labels_train = training[1]
            self.pointcloud = multidim.PointCloud.from_multisample_multilabel(pointclouds_train, labels_train)
            self.all_labels = self.pointcloud.label_info.index.values
            self.covertree = multidim.covertree.CoverTree(self.pointcloud)
        else:
            raise ValueError("bad input to fit()")

        # Do something here to make gaussians and tree.  May use self.kwargs.
        self.gaussians = self.gausscoords(**self.kwargs)
        pass

    def evaluate(self, x):
        r""" Evaluate all gaussians against a pointcloud 
        
        Parameters
        ----------
        x : :class:`numpy.ndarray`
            An array, giving a pointcloud.  Each row is a point.
       
        Returns
        -------
        functions : :class:`numpy.ndarray`
            the integral of each gaussian over these points.
        labels : :class:`numpy.ndarray`
            the labels of the gaussians, for reference.

        """

        def run_gauss(g):
            return gaussian(x, g['mean'], g['std'], g['rotation']).sum()*g['weight']
        
        functions = [run_gauss(g) for g in self.gaussians]
        labels = [g['label'] for g in self.gaussians]
        return np.array(functions), np.array(labels)

    def plot(self, canvas, style="covertree"):
        r""" Plot a CDER model, using matplotlib or bokeh

        There are several options to help visualize various aspects of the
        model.  Typically, one wants to overlay this with the plots from the
        underlying :class:`multidim.PointCloud` and 
        :class:`multidim.covertree.CoverTree` objects.

        Parameters
        ----------
        canvas : object
            An instance of 
            `bokeh.plotting.figure.Figure` as in
            :code:`canvas = bokeh.plotting.figure()`
            or an instance of :class:`matplotlib.axes._subplots.AxesSubplot` as 
            in :code:`axes,canvas = matplotlib.pyplot.subplots()`

        style : str
            options are "gaussians", "covertree", "heatmap", "hulls", 
            "expatriots", and "predictions".  Colors are generally obtained
            from the label names.

        See Also
        --------
        :func:`multidim.PointCloud.plot`
        :func:`multidim.covertree.CoverTree.plot`


        """
        if self.covertree is None or self.gaussians is None:
            raise RuntimeError("You must 'fit' first, before plotting.")

        # fix the aspect ratio!
        all_xs = self.covertree.pointcloud.coords.values[:, 0]
        all_ys = self.covertree.pointcloud.coords.values[:, 1]
        xmid = (all_xs.max() + all_xs.min())/2.0
        ymid = (all_ys.max() + all_ys.min())/2.0
        span = max([all_xs.max() - xmid,
                    xmid - all_xs.min(),
                    all_ys.max() - ymid,
                    ymid - all_ys.min()])

        if type(canvas).__module__ == 'bokeh.plotting.figure':
            canvas_type = "bokeh"
            import bokeh.plotting
            from bokeh.models import ColumnDataSource, Range1d
            
            # fix the aspect ratio!
            canvas.x_range = Range1d(xmid-span, xmid+span)
            canvas.y_range = Range1d(ymid-span, ymid+span)

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
                    "canvas must be a bokeh.plotting.figure() or a matplotlib.pyplot.subplots()[1].  You gave me {}".format(
                        type(canvas))
            )

        if style == "gaussians":
            Xms = []
            Yms = []
            R1s = []
            R2s = []
            As = []
            Cs = []
            Ws = []
            for g in self.gaussians:
                Xms.append(g['mean'][0])                                                        
                Yms.append(g['mean'][1])                                                        
                R1s.append(g['std'][0]*2/0.75)  # 0.75 bokeh oval distortion
                R2s.append(g['std'][1]*2)                                                      
                As.append(np.arctan2(g['rotation'][0, 1], g['rotation'][0, 0]))
                Cs.append(g['label'])
                Ws.append(g['weight'])
            Ws = np.array(Ws)
            # Ws = np.log2(numpy.array(Ws)+1)
            Wmax = Ws.max()
            if canvas_type == "bokeh":
                canvas.oval(Xms, Yms, angle=As, width=R1s, height=R2s,                    
                            alpha=Ws/Wmax, color=Cs, line_width=0,                    
                            line_color='black') 

            elif canvas_type == "pyplot":
                patches = []
                rgbas = []
                CC = colors.ColorConverter()
                for i in range(len(Xms)):
                    patches.append(Ellipse(xy=(Xms[i], Yms[i]),
                                           width=R1s[i]*0.75, 
                                           height=R2s[i],
                                           angle=As[i]*180/np.pi))
                    # have to manually set the alpha value
                    rgba = list(CC.to_rgba(Cs[i]))
                    rgba[3] = Ws[i]/Wmax
                    rgbas.append(tuple(rgba))

                p = PatchCollection(patches, edgecolor='none')
                p.set_facecolors(rgbas)
                canvas.add_collection(p)
            pass

        elif style == "heatmap":
            # This could probably use self.runit instead.
            xy = []
                
            min_x = xmid - span
            max_x = xmid + span
            min_y = ymid - span
            max_y = ymid + span

            dx = (max_x - min_x)/64.
            dy = (max_y - min_y)/64.
            for x in np.arange(min_x, max_x, dx):
                for y in np.arange(min_y, max_y, dy):
                    xy.append((x + dx/2, y + dy/2))
            xy = np.array(xy)
            
            z_by_label = np.ndarray(shape=(xy.shape[0], self.all_labels.shape[0]))

            for i, point in enumerate(xy):
                integrals, labels = self.evaluate(xy[[i], :])
                for j, label in enumerate(self.all_labels):
                    z_by_label[i, j] = np.linalg.norm(integrals[labels == label], 1)

            # z_by_label = np.log2(z_by_label + 1)
            # z_by_label = z_by_label - z_by_label.min()
            z_by_label = z_by_label/z_by_label.max()
            # z_by_label = z_by_label**0.5

            if canvas_type == "bokeh":
                for j, label in enumerate(self.all_labels):
                    canvas.rect(xy[:, 0]-dx/2, xy[:, 1]-dy/2, width=dx, height=dy,
                                color=label, alpha=z_by_label[:, j])
                
            elif canvas_type == "pyplot":
                patches = []
                rgbas = []
                CC = colors.ColorConverter()
                for j, label in enumerate(self.all_labels):
                    for i in range(xy.shape[0]):
                        corners = np.array([[xy[i, 0]-dx/2, xy[i, 1]-dx/2],
                                            [xy[i, 0]+dx/2, xy[i, 1]-dx/2],
                                            [xy[i, 0]+dx/2, xy[i, 1]+dy/2],
                                            [xy[i, 0]-dx/2, xy[i, 1]+dy/2]])
                        patches.append(Polygon(corners))
                        # have to manually set the alpha value
                        rgba = list(CC.to_rgba(label))
                        rgba[3] = z_by_label[i, j]
                        rgbas.append(tuple(rgba))

                p = PatchCollection(patches, edgecolor='none')
                p.set_facecolors(rgbas)
                canvas.add_collection(p)
 
            pass

        elif style == "hulls":
            from scipy.spatial import ConvexHull
            if canvas_type == "bokeh":
                raise NotImplementedError("No hulls in Bokeh yet. Use pyplot.")
            elif canvas_type == "pyplot":
                maxwt = max([g['weight'] for g in self.gaussians])
                patches = []
                rgbas = []
                cc = colors.ColorConverter()
 
                for i, g in enumerate(self.gaussians):
                    if g['count'] > 3:
                        a = g['index']
                        l = g['level']
                        cl = self.covertree[l]
                        label = g['label']
                        children = self.covertree.pointcloud.coords.values[cl.children[a], :]
                        hull = ConvexHull(children).vertices
                        poly_data = children[hull, :]
                        patches.append(Polygon(poly_data))
                        # have to manually set the alpha value
                        rgba = list(cc.to_rgba(label))
                        rgba[3] = g['weight']*1.0/maxwt
                        rgbas.append(tuple(rgba))

                p = PatchCollection(patches, edgecolor='none')
                p.set_facecolors(rgbas)
                canvas.add_collection(p)
                pass

        elif style == "entropy":
            N = len(self.covertree)
            C = np.zeros(shape=(N,))
            Q = np.zeros(shape=(N,))
            for g in self.gaussians:
                C[g['level']:] = C[g['level']:] + 1.0
                Q[g['level']:] = Q[g['level']:] + (1.0 - g['entropy'])
            canvas.scatter(range(N), C, color="red", legend="coords", size=4)
            canvas.line(range(N), C, color="red")
            canvas.scatter(range(N), Q, color="blue", legend="bits", size=2)
            canvas.line(range(N), Q, color="blue")
            canvas.legend.location = "top_left"
            pass 

        elif style == "expatriots":
            Xms = []
            Yms = []
            R1s = []
            R2s = []
            As = []
            Cs = []
            Ws = []
            for g in self.gaussians:
                Xms.append(g['mean'][0])                                                        
                Yms.append(g['mean'][1])                                                        
                R1s.append(g['std'][0]*2/0.75)  # 0.75 bokeh oval distortion
                R2s.append(g['std'][1]*2)                                                      
                As.append(np.arctan2(g['rotation'][0, 1], g['rotation'][0, 0]))
                Cs.append(g['label'])
                Ws.append(g['weight'])
            Ws = np.log2(np.array(Ws)+1)
            Wmax = Ws.max()
            if canvas_type == "bokeh":
                canvas.oval(Xms, Yms, angle=As, width=R1s, height=R2s,                    
                            alpha=Ws/Wmax, color=Cs, line_width=0,                    
                            line_color='black') 

            elif canvas_type == "pyplot":
                patches = []
                rgbas = []
                CC = colors.ColorConverter()
                for i in range(len(Xms)):
                    patches.append(Ellipse(xy=(Xms[i], Yms[i]),
                                           width=R1s[i]*0.75, 
                                           height=R2s[i],
                                           angle=As[i]*180/np.pi))
                    # have to manually set the alpha value
                    rgba = list(CC.to_rgba(Cs[i]))
                    rgba[3] = Ws[i]/Wmax
                    rgbas.append(tuple(rgba))

                p = PatchCollection(patches, edgecolor='none')
                p.set_facecolors(rgbas)
                canvas.add_collection(p)
            pass

        elif style == "predictions":
            pass
        elif style == "covertree":
            Xs = []
            Ys = []
            Rs = []
            Cs = []
            Ws = []
            for g in self.gaussians:
                Xs.append(g['adult'][0])                                                        
                Ys.append(g['adult'][1])                                                        
                Rs.append(g['radius'])                                                      
                Cs.append(g['label'])
                Ws.append(1.0 - g['entropy'])
            Ws = np.array(Ws)
            # Ws = np.log2(np.array(Ws)+1)
            Wmax = Ws.max()
            if canvas_type == "bokeh":
                canvas.circle(Xs, Ys, radius=Rs, alpha=Ws/Wmax,                   
                              color=Cs, line_width=0,                    
                              line_color='black') 
            elif canvas_type == "pyplot":
                patches = []
                rgbas = []
                cc = colors.ColorConverter()
                for i in range(len(Xs)):
                    patches.append(Circle(xy=(Xs[i], Ys[i]), radius=Rs[i]))
                    # have to manually set the alpha value
                    rgba = list(cc.to_rgba(Cs[i]))
                    rgba[3] = Ws[i]/Wmax
                    rgbas.append(tuple(rgba))

                p = PatchCollection(patches, edgecolor='none')
                p.set_facecolors(rgbas)
                canvas.add_collection(p)
            pass

        else:
            raise ValueError("Wrong plot style.")

    def predict(self, pointclouds):
        r""" 
        Predict labels of given pointclouds, based on previous training data 
        fed to :func:`fit`.

        This is just a call to :func:`runit`

        Parameters
        ----------
        pointclouds : list-like
            A list of arrays.  Each array is a pointcloud.  Each row of that
            array is a point.

        Returns
        -------
        array of labels, one for each pointcloud in the input list.
       
        """
        return self.runit(pointclouds)[2]

    def score(self, pointclouds, labels):
        r"""
        Score predicted labels against known labels.

        Parameters
        ----------
        pointclouds : list-like
            A list of arrays.  Each array is a pointcloud.  Each row of that
            array is a point.

        labels : list-like
            A corresponding list of labels.
            
        Returns
        -------
        an array of booleans.  True means the labels match.

        See Also
        --------
        :func:`score`

        """
        return self.predict(pointclouds) == labels

    def runit(self, pointclouds):

        num_tests = len(pointclouds)

        evaluations = np.ndarray(shape=(num_tests, len(self.gaussians)), dtype=np.float64)
        norm_scores = np.ndarray(shape=(num_tests, len(self.all_labels)), dtype=np.float64)
        best_match = np.ndarray(shape=(num_tests,), dtype=self.all_labels.dtype)

        # evaluate each pointcloud against each Gaussian
        for i, X in enumerate(pointclouds):
            v, l = self.evaluate(X)
            evaluations[i, :] = v
            for j, score_label in enumerate(self.all_labels):
                indicator = (l == score_label)
                norm_scores[i, j] = np.linalg.norm(v*indicator)
            best_label_index = norm_scores[i, :].argmax()
            best_match[i] = self.all_labels[best_label_index]

        return evaluations, norm_scores, best_match  


class CDER(GaussianMixtureClassifier):
    r"""The CDER (Cover-Tree Differencing for Entropy Reduction) algorithm for
    supervised machine-learning of labelled cloud collections.  This uses the
    "Cover Tree with Friends" algorithm along with an entropy computation to
    build a regional classifer for labelled pointclouds.  It relies on the
    :class:`multidim.covertree.CoverTree` and 
    :class:`multidim.covertree.CoverLevel` data structures.

    See the paper [CDER1]_ and the talk [CDER2]_.

    Parameters
    ----------

    parsimonious : bool
        Whether to use the parsimonious version of CDER, where a region is
        ignored once entropy reached a local minimum.  For most datasets,
        :code:`parsimonious=False` provides a better classifier, but they are more
        expensive to evaluate.  In many cases, :code:`parsimonious=False` is 
        good enough, and is significantly faster.
        Default: :code:`True`
    
    
    Examples
    --------
    
    We'll make a simple dataset with one "green" pointcloud sampled from 
    a uniform distribution on the 1x1 square centered at (-1,0),
    and one "magenta" pointcloud sampled from a uniform distribution on the 
    1x1 square centered at (1, 0).
    
    >>> import numpy as np
    >>> import multidim
    >>> train_dataL = np.random.rand(100,2) - np.array([-1.5, -0.5])  # for green
    >>> # dataL.mean(axis=0)  # should be near (-1, 0)
    >>> train_dataR = np.random.rand(200,2) - np.array([0.5, -0.5])  # for magenta
    >>> # dataR.mean(axis=0)  # should be near (+1, 0)
    >>> cder = CDER(parsimonious=True)  # prepare a classifier
    >>> cder.fit([train_dataL, train_dataR], ["green", "magenta"])  # this runs CoverTree
    >>> for g in cder.gaussians:
    ...     print(sorted(list(g.keys())))
    ...     break
    ['adult', 'count', 'entropy', 'index', 'label', 'level', 'mean', 'radius', 'rotation', 'std', 'weight']
    >>> test_dataL = np.random.rand(50,2) - np.array([-1.5, -0.5])  # should be green
    >>> test_dataR = np.random.rand(50,2) - np.array([0.5, -0.5])  # should be magenta
    >>> cder.predict([test_dataL, test_dataR])  # Guess the labels
    array(['green', 'magenta'], dtype=object)
    >>> cder.score([test_dataL, test_dataR], ["green", "magenta"])  # Correct?
    array([ True,  True], dtype=bool)
    

    A more thorough example is at http://nbviewer.jupyter.org/github/geomdata/gda-public/blob/master/examples/example-cder.ipynb

    See Also
    --------
    :class:`multidim.covertree.CoverTree`

    References
    ----------
    .. [CDER1] Supervised Learning of Labeled Pointcloud Differences via Cover-Tree Entropy Reduction https://arxiv.org/abs/1702.07959
    .. [CDER2] CDER, Learning with Friends https://www.ima.umn.edu/2016-2017/DSS9.6.16-5.30.17/26150

    """

    def build_gaussian(self, coverlevel, adult, label, dominant_index):
        r"""
        Build a Gaussian from the children of this adult in a cover-tree
        (using only a particular label).

        Generally, a user will never call this.  It is called when
        :func:`gausscoords` decides that a CoverTree ball is work modelling.

        Parameters
        ----------
        coverlevel : `multidim.covertree.CoverLevel`
        adult : int
        label : int
        dominant_index : int

        See Also
        --------
        :func:`gausscoords`

        """
        ct = self.covertree
        pc = ct.pointcloud
        label_index = pc.label_info['int_index'].loc[label]
        assert label == pc.label_info.index[label_index]
        winning_children = np.intersect1d(coverlevel.children[adult], np.where(pc.labels == label_index ))
        count = len(winning_children)
        sample = ct.coords[winning_children,:]
        entropy = coverlevel.entropy[adult]
        ambient_dim = ct.coords.shape[1]
        weight = coverlevel.weights[adult][label_index].sum()
        if count < ambient_dim:
            logging.warn("Too few points {} at {}.  Lost {}*{}".format(count, adult, 1-entropy, weight))
            return None
        m,s,v = gaussian_fit(sample)
        if np.any(s <= 0.):
            logging.warn("Bad Gaussian {} at {}.  Lost {}*{}".format(s, adult, 1-entropy, weight))
            return None
        return {"label": label,
                "mean": m,
                "std": s,
                "rotation": v,
                "index": adult,
                "adult": ct.coords[adult,:],
                "level": coverlevel.exponent,
                "radius": coverlevel.radius,
                "entropy": entropy,
                "weight": coverlevel.radius**ambient_dim*weight*(1. - entropy ),
                "count":  count,
                }
   

    def gausscoords(self, parsimonious=True):
        r""" This is the Gaussian-building heart of the CDER algorithm. 
        Look to elders, and look to successor.  How does entropy change?   If
        the entropy is a local minimum, then build a Gaussian coordinate with
        the dominant label.
        This populates :code:`self.gaussians` for a :class:`CDER` object.

        Generally, a user will never call this.  It is called by
        :func:`fit`

        Parameters
        ----------
        parsimonious:  bool
            If True, then ignore unlikely regions quickly, to minimize the number of
            Gaussian coordinates.  If False, then check all levels of the
            CoverTree.  (default: True)

        See Also
        --------
        :func:`build_gaussians` :func:`fit`
        """
    
        gaussians = []
        for next_level in self.covertree:
            # next_level is *not* the level we are actually studying!
            # but, we need to pre-compute to see if it is worth proceeding.
            if next_level.exponent > 16:
                break
            elif next_level.exponent <= 1: 
                next_level.adults_to_check = []
                next_level.adults_to_check.extend(next_level.adults)
                for adult in next_level.adults_to_check:
                    next_level.find_entropy(adult)
                pass
            else:
                l_plus_1 = next_level.exponent
                prev_level = self.covertree._levels[l_plus_1-2]
                this_level = self.covertree._levels[l_plus_1-1]
                
                if not this_level.adults_to_check: 
                    logging.info("Done at {}".format(this_level.exponent))
                    break # done with covertree!
                
                next_level.adults_to_check = []
                logging.info("Gaussians so far: {}".format(len(gaussians)))
                logging.info(this_level)
                logging.info("Checking {} adults".format(len(this_level.adults_to_check)))
                for adult in this_level.adults_to_check:
                    
                    # TODO -- make this more efficient with special ratio.
                    # get union of entropy of elders
                    adult_a = np.array([adult], dtype=np.int64)
                    pre_elders = np.array(prev_level.friends1[this_level.predecessor[adult]])
                    pre_eldersR = distance_cache_None(
                        adult_a, pre_elders, self.covertree.coords).flatten()
                    my_elders = pre_elders[pre_eldersR <= prev_level.radius]
                    prev_weights = np.concatenate([ prev_level.weights[e] for e in my_elders ])
                    
                    prev_entropy = entropy(prev_weights/prev_weights.sum(axis=0))
                    assert not np.isnan(prev_entropy), "nan?  prev_weights = {}".format(prev_weights)
                    # union entropy of elders of adult at prev_level
                    this_entropy = this_level.find_entropy(adult)
                    assert not np.isnan(this_entropy)
                    # entropy of this_level children of adult
                    next_entropy = next_level.find_entropy(adult)
                    assert not np.isnan(next_entropy)

                    logging.debug("{}: {} {} {}:".format(adult, next_entropy, this_entropy, prev_entropy), end=": ")
                    

                    # entropy of next_level children of adult
                    # We consider all orderings of these three quantities
                    if np.count_nonzero(next_level.children[adult]) <= this_level.covertree.pointcloud.multiplicity[adult]:
                        pass
                    elif np.count_nonzero(this_level.children[adult]) <= this_level.covertree.pointcloud.multiplicity[adult]:
                        pass
                    elif  1.0 > prev_entropy >= this_entropy >= next_entropy:
                        # use it!
                        my_weights = this_level.weights[adult]
                        totweight = my_weights.sum()
                        numlabels = my_weights.shape[0]
        
                        # find the labels that dominate the low-entropy
                        # sort from biggest to least, and make a coord for each.
                        dominant_index = (my_weights > totweight/numlabels)
                        dominant_labels = this_level.covertree.pointcloud.label_info.index[dominant_index]
                        label_ordering = my_weights[dominant_index].argsort()[::-1]
                        for label in dominant_labels[label_ordering]:
                                g = self.build_gaussian(this_level, adult, label, dominant_index)
                                if g is not None:
                                    gaussians.append(g)
                        if parsimonious is False:
                            to_check = list(this_level.successors[adult])
                            next_level.adults_to_check.extend(to_check)
                    
                    elif 1.0 > this_entropy >= prev_entropy >= next_entropy:
                        # re-check adult at the next level.
                        if parsimonious is False:
                            to_check = list(this_level.successors[adult])
                            next_level.adults_to_check.extend(to_check)
                        elif parsimonious is True:
                            # only the center is likely to be useful
                            next_level.adults_to_check.append(adult)
                        else:
                            raise ValueError("parsimonious flag must be True or False.")
                    
                    elif 1.0 > prev_entropy >= next_entropy >= this_entropy:
                        to_check = list(this_level.successors[adult])
                        if parsimonious is True:
                            # center is unlikely to be useful.
                            to_check.remove(adult)
                        next_level.adults_to_check.extend(to_check)
    
                    elif 1.0 > next_entropy >= prev_entropy >= this_entropy:
                        to_check = list(this_level.successors[adult])
                        if parsimonious is True:
                            # center is unlikely to be useful.
                            to_check.remove(adult)
                        next_level.adults_to_check.extend(to_check)
    
                    elif 1.0 > this_entropy >= next_entropy >= prev_entropy:
                        if parsimonious is False:
                            to_check = list(this_level.successors[adult])
                            next_level.adults_to_check.extend(to_check)
                        pass # do nothing with this subtree.  That is, STOP.
                    
                    elif 1.0 > next_entropy >= this_entropy >= prev_entropy:
                        if parsimonious is False:
                            to_check = list(this_level.successors[adult])
                            next_level.adults_to_check.extend(to_check)
                        pass # do nothing with this subtree.  That is, STOP.
                    
                    else:
                        assert max([prev_entropy, this_entropy, next_entropy]) >= 1.0, "I see an strange entropy value {}".format([prev_entropy, this_entropy, next_entropy])
                        to_check = list(this_level.successors[adult])
                        next_level.adults_to_check.extend(to_check)
    
        return gaussians
