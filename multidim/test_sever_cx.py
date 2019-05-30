r"""
Test until_connected on a simplicial complex by persistence.

Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE

>>> import numpy as np
>>> import pandas as pd
>>> from scipy.spatial import distance_matrix
>>> from multidim import SimplicialComplex
>>> data = np.random.uniform(0, 10, 20).reshape(-1, 2)
>>> edm = distance_matrix(data, data)
>>> pc = SimplicialComplex.from_distances(edm)
>>> pc.make_pers0(cutoff=2.0)
>>> count = 0
>>> for sub_index, sub_pc in pc.sever():
...     count += len(sub_pc.stratum[0])
>>> assert count == 10
"""
