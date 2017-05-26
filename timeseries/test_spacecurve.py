r"""
Basic tests of data access and computation for `SpaceCurve`

Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE

Examples
--------

>>> from __future__ import print_function
>>> from timeseries import SpaceCurve, curve_geometry
>>> F = SpaceCurve.load("tests/spiral.csv")
>>> F
SpaceCurve with 126 entries and duration 625.000000000
>>> for label in F.data.columns:
...     print(label)
time
pos_x
pos_y
pos_z
quality
>>> F[0]
time       0.0
pos_x      1.0
pos_y      0.0
pos_z      0.0
quality   -1.0
Name: 0, dtype: float64
>>> time = F.data['time'].values
>>> pos = F.data[['pos_x','pos_y','pos_z']].values
>>> F.compute()
>>> vel = F.info[['vel_x','vel_y','vel_z']].values
>>> arclengthS = curve_geometry.secant_arclength(pos)


"""

