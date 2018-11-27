r""" 
Test barycenter code for :class:`multidim.PointCloud`

Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE

"""

import numpy as np
import multidim
NDBL = np.float64


class TestBarycenters:

    def setup(self):
        X = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        self.PC = multidim.PointCloud(X)

    def teardown(self):
        del self.PC

    def setup_method(self, f):
        pass
    
    def teardown_method(self, f):
        pass

    def test_knn(self):
        self.PC.nearest_neighbors(2)
        assert np.all(self.PC.nearest_neighbors(2) == np.array([
            [0, 1, 3],
            [1, 0, 2],
            [2, 1, 3],
            [3, 0, 2]]))

    def test_bary(self):
        wb = self.PC.witnessed_barycenters(3)
        assert np.all(wb.stratum[0]['height'].values*3. == 2.)

    def test_rand(self):
        randX = np.random.rand(1000, 2)
        randPC = multidim.PointCloud(randX)
        randPC.witnessed_barycenters(3)

if __name__ == '__main__':
    T = TestBarycenters()
    T.setup()
    T.test_knn()
    T.test_bary()
    T.test_rand()
    T.teardown()
