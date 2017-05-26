r"""
Test the mollifier code

Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE
"""

import numpy
from timeseries import curve_geometry

class TestMollifier:
    def setup(self):
        a = -1.0
        b = 1.0
        h = 0.3
        m = 100
        sigma = .1
        numpy.random.seed(seed=70713)

        # set up original function and noisy version
        self.dt = (b - a)/m
        self.time = a + self.dt * numpy.arange(m)
        error = numpy.random.uniform(low=-1.0, high=1.0, size=m)*sigma
        fWithNoise = numpy.cos(self.time) + error
        fWithNoise[int(m / 2)] = 1.4
        self.position = fWithNoise
        self.num_refinements = 3


    def teardown(self):
        del self.time
        del self.dt
        del self.position
        del self.num_refinements

    def setup_method(self, function):
        pass

    def teardown_method(self, function):
        pass

    def test_refinement_counts(self):
        point_count = self.time.shape[0]
        for refinement in range(self.num_refinements):
            refined = curve_geometry.mollifier(self.time, self.position,
                                               (refinement+1), width=0.3)

            assert (refinement+1)*point_count==refined[0,:].shape[0],\
            """input time shape {} and refined shape {} are mismatched
            for refinement {}.""".format(self.time.shape[0],
                                         refined[0,:].shape[0],
                                         (refinement+1))


    def test_resonable_reconstruction(self):
        for refinement in range(self.num_refinements):
            R = curve_geometry.mollifier(self.time, self.position,
                                         2**(refinement+1), width=0.3)
            # \|cos(t)\|_\infty = 1 \implies rel_err = abs_err
            err = numpy.linalg.norm((numpy.cos(R[0,:])-R[1,:]),ord=numpy.inf)
            assert err <= 0.1,\
            """Failed to make a resonable reconstruction,
            relative error {} > 0.1""".format(err)

if __name__ == '__main__':
    T = TestMollifier()
    T.setup()
    T.test_refinement_counts()
    T.teardown()
