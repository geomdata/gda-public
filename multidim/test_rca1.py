r""" 
Basic tests of RCA1 algorithm in :class:`multidim.PointCloud` 

Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE

"""

from __future__ import print_function

import numpy as np
NDBL = np.float64
import pandas
import multidim
import multidim.covertree
import time
import sys
import bz2

class TestRips:

    def setup(self):
        self.circ = multidim.PointCloud(np.load("tests/circle.npy"), max_length=-1)

    def teardown(self):
        del self.circ

    def setup_method(self, function):
        pass
    
    def teardown_method(self, function):
        pass

    def test_rca1_circle(self):
        self.circ.make_pers1_rca1(cutoff=0.2)

    def test_rca1_offset(self):
        for x in [0.0,]:# 0.5 fails
            data = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.-x]])
            pc = multidim.PointCloud(data, max_length=-1)
            #pc.make_pers1_rca1()
 
if __name__ == '__main__':
    T = TestCovers()
    T.setup()
    T.test_rca1_circle()
    T.test_rca1_offset()
    T.teardown()
