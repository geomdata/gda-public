r"""
Test until_connected on a Signal by persistence.

Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2019 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE

>>> import numpy as np
>>> import timeseries
>>> from homology.dim0 import all_roots
>>> t = np.linspace(-4*np.pi,4*np.pi,20)
>>> S = timeseries.Signal(t*(np.sin(t)))
>>> # fig,ax = plt.subplots()
>>> # S.plot(ax)
>>> S.make_pers(until_connected=(6,13))
>>> S.pers.diagram
   birth_index  death_index      birth     death       pers
0            9            8   0.406233  1.817044   1.410811
1           13           11  -4.613903  1.817044   6.430947
2            1            3 -10.899544  6.325786  17.225330
>>> roots = S.components.values.copy()
>>> all_roots(roots)
>>> #plt.scatter(S.vertices['time'], S.vertices['time'].loc[roots])
>>> print(roots)
[ 1  1  1  3  4  6  6  6  6  6  6  6  6  6  6 15 16 18 18 18]
"""
