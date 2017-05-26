r""" 
Basic tests of CDER

Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE

"""

import sys

import numpy 

import time

from multidim.models import CDER
import multidim
from multidim.covertree import CoverTree


list_of_pointclouds = []
list_of_labels = []
for i in range(50):
    R = numpy.concatenate([numpy.random.randn(20,2), 0.2*numpy.random.randn(1,2) + numpy.array([[3,3]])])
    list_of_pointclouds.append(R)
    list_of_labels.append("red")
    B = numpy.concatenate([numpy.random.randn(20,2), 0.2*numpy.random.randn(1,2) + numpy.array([[-1,-1]])])
    list_of_pointclouds.append(B)
    list_of_labels.append("blue")

list_of_pointclouds = numpy.array(list_of_pointclouds)
list_of_labels = numpy.array(list_of_labels)

#print("forming CoverTree")
#pc = multidim.multisample_multilabel_pointcloud(list_of_pointclouds, list_of_labels)
#ct = CoverTree(pc)


# here's how we would break up stuff manually, 
#pointclouds_train, pointclouds_test, labels_train, labels_test = cross_validation.train_test_split(pointclouds, labels, test_size=0.2, random_state=0)

# but, we can also just do this:
#print("cross-validating")
cder = CDER() #stop_level=5)
cder.fit(list_of_pointclouds, list_of_labels)

#xx = cross_validation.cross_val_score(gcte, pointclouds, y=labels,
#    scoring='accuracy',
#    verbose=2,
#    n_jobs=-1,
#    cv=5)
#print(xx.sum()*100.0/len(xx))


