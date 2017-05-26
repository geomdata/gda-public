r"""
Cross-validation test of CDER


Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE


>>> import numpy as np
>>> from sklearn.model_selection import cross_val_score, train_test_split
>>> import multidim
>>> from multidim.models import CDER
>>> 
>>> pointclouds = []
>>> labels = []
>>> for i in range(100):
...     R = np.concatenate([np.random.randn(20,2), 0.2*np.random.randn(2,2) + np.array([[3,3]])])
...     pointclouds.append(R)
...     labels.append('red')
...     B = np.concatenate([np.random.randn(20,2), 0.2*np.random.randn(2,2) + np.array([[-3,-3]])])
...     pointclouds.append(B)
...     labels.append('blue')
>>>
>>> pointclouds = np.array(pointclouds)
>>> print(pointclouds.shape)
(200, 22, 2)
>>> labels = np.array(labels)
>>> pointclouds_train, pointclouds_test, labels_train, labels_test = train_test_split(pointclouds, labels, test_size=0.2, random_state=0)
>>> 
>>> cder = CDER(parsimonious=True)
>>> cder.fit(pointclouds_train, labels_train)
>>> 
>>> assert len(cder.gaussians) < 50, "Too many Gaussians?"
>>> 
>>> #fig=figure()
>>> #cder.pointcloud.plot(fig)
>>> #show(fig)
>>> #fig=figure()
>>> #cder.plot(fig, style="gaussians")
>>> #show(fig)
>>> #fig=figure()
>>> #cder.plot(fig, style="heatmap")
>>> #show(fig)
>>> #fig = figure( title="bits of certainty per level")
>>> #cder.plot(fig, style="entropy")
>>> #fig.legend.location="top_left"
>>> #show(fig)
>>> 
>>> cder = CDER(parsimonious=True)
>>> results = cross_val_score(cder, pointclouds, labels, scoring='accuracy', cv=5)
>>> assert results.mean() <= 1.0, "CDER cross_validation is broken."
>>> assert 0.80 <= results.mean(), "CDER is less accurate than I expect! {}".format(results.mean())
"""
