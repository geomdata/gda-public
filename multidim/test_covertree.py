r""" 
Basic tests of :class:`multidim.covertree.CoverTree`

Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE

"""

from __future__ import print_function

import numpy
NDBL = numpy.float64
import pandas
import multidim
import multidim.covertree
import time
import sys
import bz2
class TestCovers:

    def setup(self):
        self.PC = multidim.PointCloud(numpy.load("tests/circle.npy"))

    def teardown(self):
        del self.PC

    def setup_method(self, function):
        pass
    
    def teardown_method(self, function):
        pass

    def test_covers(self):
        t0=time.clock()
        ct = multidim.covertree.CoverTree(self.PC, ratio=0.5, exchange_teens=False,
            sort_orphans_by_mean=False)
        s = ""
        for cl in ct:
            cl.check()
            s = s+"{}\n".format(cl)
            s = s+"{}\n".format(sorted(list(cl.adults)))
            for ci in sorted(list(cl.adults)):
                s = s+"{} F1 {}\n".format(ci, cl.friends1[ci])
                s = s+"{} F2 {}\n".format(ci, cl.friends2[ci])
                s = s+"{} F3 {}\n".format(ci, cl.friends3[ci])
        t1 = time.clock()
        assert t1-t0 < 30.0, "Took too long!"
        if sys.version_info[0] >= 3 and sys.version_info[1] >= 3:
            with bz2.open("tests/circle-covertree-verified.txt.bz2", "rt") as f:
                assert s == f.read(),\
                    "CoverTree output did not match verified(mostly) example."
        else:
            print("Skipping test due to python version. {}".format(sys.version_info))

    def test_covers_exchange(self):
        t0=time.clock()
        ct = multidim.covertree.CoverTree(self.PC, ratio=0.5, exchange_teens=True)
        s = ""
        for cl in ct:
            #print(cl)
            assert cl.check()
            #s = s+"{}\n".format(cl)
            #s = s+"{}\n".format(sorted(list(cl.adults)))
            #for ci in sorted(list(cl.adults)):
            #    s = s+"{} F1 {}\n".format(ci, cl.friends1[ci])
            #    s = s+"{} F2 {}\n".format(ci, cl.friends2[ci])
            #    s = s+"{} F3 {}\n".format(ci, cl.friends3[ci])
        t1 = time.clock()
        assert t1-t0 < 30.0, "Took too long!"
        #with bz2.open("tests/circle-covertree-verified.txt.bz2", "rt") as f:
        #    assert s == f.read(),\
        #        "CoverTree output did not match verified(mostly) example."


    #def test_sparse_complex(self):
    #    for level in self.CT2:
    #        pass
    #    sc = self.C2.sparse_complex()
    #    assert len(sc.stratum[0]) == len(self.CT2.centers)
    #    assert len(sc.stratum[1]) == 4
    #    print("INCOMPLETE TEST!  No Check for Correctness Implemented.", end=" ")

if __name__ == '__main__':
    T = TestCovers()
    T.setup()
    T.test_covers_reassign()
    #T.test_covers_compare_simple()
    #T.test_sparse_complex()
    T.teardown()
