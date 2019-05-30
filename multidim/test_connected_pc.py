r"""
Test until_connected on a PointCloud by persistence.

Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2019 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE

>>> import numpy as np
>>> from multidim import PointCloud
>>> pc = PointCloud(np.array([
...      [  -3.0,  0.],
...      [   0.0,  4.0],
...      [   1.0,  4.0],
...      [   6.5,  4.0],
...      [   9.5,  0.],
...      ]), max_length=-1.0)
>>> pc.stratum[0]
   height  mass   pos  rep
0     0.0   1.0  True    0
1     0.0   1.0  True    1
2     0.0   1.0  True    2
3     0.0   1.0  True    3
4     0.0   1.0  True    4
>>> pc.stratum[1]
      height   pos  rep  bdy0  bdy1
0   1.000000  True    0     1     2
1   5.000000  True    1     0     1
2   5.000000  True    2     3     4
3   5.500000  True    3     2     3
4   5.656854  True    4     0     2
5   6.500000  True    5     1     3
6   9.394147  True    6     2     4
7  10.307764  True    7     0     3
8  10.307764  True    8     1     4
9  12.500000  True    9     0     4
>>> pc.make_pers0(until_connected=(0,4))
>>> pc.pers0.diagram['death'].max()
5.5
>>> roots = pc.stratum[0]['rep'].values.copy()
>>> from homology.dim0 import all_roots
>>> all_roots(roots)
>>> roots
array([0, 0, 0, 0, 0])
"""
