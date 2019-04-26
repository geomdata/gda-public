r"""
Test severing a PointCloud by persistence.

Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE

>>> import numpy as np
>>> from multidim import PointCloud
>>> pc = PointCloud(np.array([[ 179.,  695.],
...      [ 181.,  693.],
...      [ 182.,  691.],
...      [ 185.,  701.],
...      [ 187.,  692.],
...      [ 188.,  704.],
...      [ 191.,  696.],
...      [ 193.,  698.],
...      [ 195.,  701.],
...      [ 195.,  710.],
...      [ 197.,  702.],
...      [ 197.,  711.],
...      [ 199.,  706.],
...      [ 662.,  542.],
...      [ 663.,  545.],
...      [ 663.,  548.],
...      [ 665.,  542.],
...      [ 667.,  549.],
...      [ 667.,  551.],
...      [ 667.,  563.],
...      [ 669.,  543.],
...      [ 669.,  551.],
...      [ 670.,  546.],
...      [ 670.,  560.],
...      [ 671.,  557.]]), max_length=-1.0)
>>> pc.make_pers0(400.)
>>> for sub_index, sub_pc in pc.sever():
...     print(sub_pc)
...     print(sub_pc.coords)
A SimplicialComplex with 13 points, 78 edges, and 0 faces.
        0      1
0   179.0  695.0
1   181.0  693.0
2   182.0  691.0
3   185.0  701.0
4   187.0  692.0
5   188.0  704.0
6   191.0  696.0
7   193.0  698.0
8   195.0  701.0
9   195.0  710.0
10  197.0  702.0
11  197.0  711.0
12  199.0  706.0
A SimplicialComplex with 12 points, 66 edges, and 0 faces.
        0      1
13  662.0  542.0
14  663.0  545.0
15  663.0  548.0
16  665.0  542.0
17  667.0  549.0
18  667.0  551.0
19  667.0  563.0
20  669.0  543.0
21  669.0  551.0
22  670.0  546.0
23  670.0  560.0
24  671.0  557.0
"""
