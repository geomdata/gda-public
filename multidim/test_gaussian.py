r""" 
Basic tests of gaussian evaluation code

Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE


>>> import numpy, multidim
>>> import multidim.fast_algorithms as tda
>>> tda.gaussian(numpy.array([[0.]]), numpy.array([0.]), numpy.array([1.]), numpy.array([[1.]]))[0] == 1./numpy.sqrt((2*numpy.pi))
True
>>> tda.gaussian(numpy.array([[1.]]), numpy.array([0.]), numpy.array([1.]), numpy.array([[1.]]))[0] == 1./numpy.sqrt((2*numpy.pi))*numpy.exp(-0.5)
True
>>> tda.gaussian(numpy.array([[0.5]]), numpy.array([1.]), numpy.array([0.5]), numpy.array([[1.]]))[0] == 1./0.5/numpy.sqrt((2*numpy.pi))*numpy.exp(-0.5)
True
>>> data = numpy.array([[0.,0.], [1.,0.], [-1.,0.], [0.,2.], [0.,-2.]])
>>> PC = multidim.PointCloud(data)
>>> mean, s, v = PC.gaussian_fit()
>>> print(mean)
[ 0.  0.]
>>> numpy.allclose(s, numpy.sqrt(numpy.array([ 2.0, 0.5])))
True
>>> K=1./numpy.sqrt((2*numpy.pi)**2)
>>> sample_point = numpy.array([mean])
>>> prob = tda.gaussian(sample_point, mean, s, v)
>>> S=numpy.abs(s.prod())
>>> numpy.allclose(prob[0], K/S)
True
>>> data = numpy.array([[25., 15.], [-15., -15.], [3.2, 2.4]])
>>> PC = multidim.PointCloud(data)
>>> mean, s, v = PC.gaussian_fit()
>>> print(mean)
[ 4.4  0.8]
>>> numpy.allclose(s**2, numpy.array([625.0, 3.0]))
True
>>> K=1./numpy.sqrt((2*numpy.pi)**2)
>>> sample_point = numpy.array([mean])
>>> prob = tda.gaussian(sample_point, mean, s, v)
>>> S=numpy.abs(s.prod())
>>> numpy.allclose(prob[0], K/S)
True
>>> data = numpy.array([[17.,-10.], [-15., 14.],[1., 2.], [43.,58.], [-41.,-54.]])
>>> PC = multidim.PointCloud(data)
>>> mean, s, v = PC.gaussian_fit()
>>> print(mean)
[ 1.  2.]
>>> numpy.allclose(s**2, numpy.array([2450.0, 200.0]))
True
>>> K=1./numpy.sqrt((2*numpy.pi)**2)
>>> sample_point = numpy.array([mean])
>>> prob = tda.gaussian(sample_point, mean, s, v)
>>> S=numpy.abs(s.prod())
>>> numpy.allclose(prob[0], K/S)
True
"""
