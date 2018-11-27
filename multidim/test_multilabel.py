r"""
Test weighting and labeling of PointClouds


Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE


>>> import numpy, multidim
>>> # A "generic" example, with mutliple samples and multiple labels
>>> list_of_samples = [ 
...     numpy.array([[0.0, 0.0], [0.0, 1.0]]),
...     numpy.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]),
...     numpy.array([[2.0, 0.0], [2.0, 1.0], [2.0, 2.0]])]
>>> list_of_labels = [ "red", "blue", "red" ]
>>> pc = multidim.PointCloud.from_multisample_multilabel(list_of_samples, list_of_labels)
>>> print(pc.stratum[0])
   height      mass   pos  rep
0     0.0  0.250000  True    0
1     0.0  0.250000  True    1
2     0.0  0.333333  True    2
3     0.0  0.333333  True    3
4     0.0  0.333333  True    4
5     0.0  0.166667  True    5
6     0.0  0.166667  True    6
7     0.0  0.166667  True    7
>>> print(pc.label_info)
      clouds  points  weight  int_index
blue       1       3     1.0          0
red        2       5     1.0          1
>>> print(pc.labels)
[1 1 0 0 0 1 1 1]
>>> print(pc.label_info.iloc[pc.labels].index.values)
['red' 'red' 'blue' 'blue' 'blue' 'red' 'red' 'red']
>>> print(pc.source)
[0 0 1 1 1 2 2 2]
>>> print(pc.coords)
     0    1
0  0.0  0.0
1  0.0  1.0
2  1.0  0.0
3  1.0  1.0
4  1.0  2.0
5  2.0  0.0
6  2.0  1.0
7  2.0  2.0
>>> # Let's run CoverTree, to see what happend
>>> from multidim.covertree import CoverTree
>>> ct = CoverTree(pc)
>>> for cl in ct:
...     print(cl)
...     print(cl.adults)
...     for a in cl.adults:
...         print(a)
...         print(cl.weights[a])
Level 0 using 1 adults at radius 1.414213562...
[3]
3
[ 1.  1.]
Level 1 using 8 adults at radius 0.585786437...
[3, 1, 2, 4, 6, 0, 5, 7]
3
[ 0.33333333  0.        ]
1
[ 0.    0.25]
2
[ 0.33333333  0.        ]
4
[ 0.33333333  0.        ]
6
[ 0.          0.16666667]
0
[ 0.    0.25]
5
[ 0.          0.16666667]
7
[ 0.          0.16666667]
>>> # Here is an example with only one sample and only one label.
>>> list_of_samples = [numpy.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])]
>>> list_of_labels = [ 'red' ]
>>> pc = multidim.PointCloud.from_multisample_multilabel(list_of_samples, list_of_labels)
>>> pc.label_info
     clouds  points  weight  int_index
red       1       3     1.0          0
>>> print(pc.stratum[0])
   height      mass   pos  rep
0     0.0  0.333333  True    0
1     0.0  0.333333  True    1
2     0.0  0.333333  True    2
>>> print(pc.labels)
[0 0 0]
>>> print(pc.source)
[0 0 0]
>>> print(pc.coords)
     0    1
0  0.0  0.0
1  0.0  1.0
2  0.0  2.0
>>> # Here is the default behavior, when no label is given.
>>> pc = multidim.PointCloud(numpy.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]),
...     masses = numpy.array([1.0, 1.0, 1.0]))
>>> pc.label_info
       clouds  points  tot_mass  int_index
black       1       3       3.0          0
>>> print(pc.stratum[0])
   height  mass   pos  rep
0     0.0   1.0  True    0
1     0.0   1.0  True    1
2     0.0   1.0  True    2
>>> print(pc.labels)
[0 0 0]
>>> print(pc.source)
[0 0 0]
>>> print(pc.coords)
     0    1
0  0.0  0.0
1  0.0  1.0
2  0.0  2.0
"""
