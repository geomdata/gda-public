r"""
Test lower-star filtration of a rectangular grid ("image")


Copyright
---------
- This file is part of https://github.com/geomdata/gda-public/ 
- 2015, 2016, 2017 by Geometric Data Analytics, Inc. (http://geomdata.com)
- AGPL license. See `LICENSE` or https://github.com/geomdata/gda-public/blob/master/LICENSE
"""

import numpy
import multidim


class TestLowerStar:
    r"""
    This is a class for testing lower star on simplicial image grid where the
    image pixels are vertices and weight is determined by their value.

    Notes
    -----
    One can calculate the *star*, *link*, *lower star*, and *lower link* for
    the image numpy.array([[1, 2], [4, 3]]) represented as an simplicial
    complex. This input corresponds to the poset where the vertices
    are weighted by the input.

    .. math::

        K = \Big \{\emptyset, \{0\}, \{1\}, \{2\},
        \{3\}, \{0,1\}, \{2,3\}, \{0,2\}, \{1,3\}, \{1,2\},
        \{0,1,2\}, \{1,2,3\} \Big \}

      * **Star:** Let :math:`K` be an abstract simplicial complex.
      Let :math:`v` be a vertex. The *star* (open star)
      of :math:`v` is the set of simplices which contain :math:`v`:

      .. math::

          St(v) = \{\sigma \in K \mid v \in \sigma\}

      For our example we have the set of vertices :math:`V = \{0,1,2,3\}` as
      labeled in diagram

      .. math::

          \begin{aligned}
          St(0) &= \Big \{ \{0\}, \{0, 1\}, \{0, 2\}, \{0, 1, 2\}
          \Big \} \\
          St(1) &= \Big \{ \{1\}, \{0, 1\}, \{1, 3\}, \{1, 2\}, \{0, 1, 2\},
          \{1, 2, 3\} \Big \} \\
          St(2) &= \Big \{\{2\}, \{2, 3\}, \{0, 2\}, \{1, 2\}, \{0, 1, 2\},
          \{1, 2, 3\} \Big \} \\
          St(3) &= \Big \{\{3\}, \{2, 3\}, \{1, 3\}, \{1, 2, 3\} \Big \}.
          \end{aligned}

      * **Lower star:** let :math:`K` and :math:`V` be as above. The
      *lower star* of :math:`v \in V` is

      .. math::

           St_{-}(v) = \{\sigma \in St(v) \mid f(x) \leq f(v),
           \forall x \in \sigma\}.

      In our case we calculate the lower stars for :math:`v \in V`

      .. math::

          \begin{aligned}
          St_{-}(0) &= \Big \{\{0\} \Big\} \\
          St_{-}(1) &= \Big \{\{1\}, \{0, 1\} \Big \} \\
          St_{-}(2) &= \Big \{\{2\}, \{2,3\}, \{0,2\}, \{1,2\},
          \{0,1,2\}, \{1,2,3\} \Big \} \\
          St_{-}(3) &= \Big \{\{3\}, \{1, 3\} \Big \}.
          \end{aligned}
    """
    def setup(self):
        A = numpy.array([[1, 2], [4, 3]])
        self.lowerStar = multidim.lower_star_for_image(A)

        # build poset representation

        # vertices with values
        self.vertices = []
        for cell in self.lowerStar.cells(0):
            self.vertices += [[cell.index,cell.height]]

        # print(self.vertices)

        # edges with values
        self.edges = []
        for cell in self.lowerStar.cells(1):
            self.edges += [[cell.boundary,cell.height]]

        # print(self.edges)

        # faces with values
        self.faces = []
        for cell in self.lowerStar.cells(2):
            boundary = cell.boundary
            face = []
            for item in boundary:
                face += list((self.edges[item])[0])
            self.faces += [[set(face),cell.height]]

        # print(self.faces)

        # build the stars

        self.stars = {}
        for vertex in self.vertices:
            star = []
            # find 0 dimensional simplicies having 'vertex'
            for simp in self.vertices:
                if simp[0] == vertex[0]:
                    star += [simp[0]]
            # find 1 dimensional simplicies having 'vertex'
            for simp in self.edges:
                if vertex[0] in simp[0]:
                    star += [simp[0]]
            # find 2 dimensional simplicies having 'vertex'
            for simp in self.faces:
                if vertex[0] in simp[0]:
                    star += [simp[0]]
            self.stars['St('+str(vertex[0])+')'] = star

        # print(self.stars)

        # build the lower-stars

        self.lower_stars = {}
        for vertex in self.vertices:
            low_star = []
            # check vertex weight against function on 0d simplices
            for simp in self.vertices:
                if simp[1] == vertex[1]:
                    low_star += [simp[0]]
            # check vertex weight against function on 1d simplices
            for simp in self.edges:
                if simp[1] == vertex[1]:
                    low_star += [simp[0]]
            # check vertex weight against function on 2d simplices
            for simp in self.faces:
                if simp[1] == vertex[1]:
                    low_star += [simp[0]]
            self.lower_stars['St_{-}('+str(vertex[0])+')'] = low_star

        # print(self.lower_stars)

        pass

    def teardown(self):
        del self.lowerStar
        del self.vertices
        del self.edges
        del self.faces
        del self.stars
        del self.lower_stars
        pass

    def setup_method(self, function):
        pass

    def teardown_method(self, function):
        pass

    def test_star(self):
        # test St(0)
        assert self.stars['St(0)'] == [0, {0, 1}, {0, 2}, {0, 1, 2}],\
            """St(0) is incorrect"""
        # test St(1)
        assert self.stars['St(1)'] == \
            [1, {0, 1}, {1, 3}, {1, 2}, {0, 1, 2}, {1, 2, 3}], \
            """St(1) is incorrect"""
        # test St(2)
        assert self.stars['St(2)'] == \
            [2, {2, 3}, {0, 2}, {1, 2}, {0, 1, 2}, {1, 2, 3}], \
            """St(2) is incorrect"""
        # test St(3)
        assert self.stars['St(3)'] == [3, {2, 3}, {1, 3}, {1, 2, 3}],\
            """St(3) is incorrect"""
        pass

    def test_lower_star(self):
        assert self.lower_stars['St_{-}(0)'] == [0],\
            """St_{-}(0) is incorrect"""
        assert self.lower_stars['St_{-}(1)'] == [1, {0, 1}],\
            """St_{-}(1) is incorrect"""
        assert self.lower_stars['St_{-}(2)'] == \
               [2, {2, 3}, {0, 2}, {1, 2}, {0, 1, 2}, {1, 2, 3}],\
               """St_{-}(2) is incorrect"""
        assert self.lower_stars['St_{-}(3)'] == [3, {1, 3}],\
            """[3, {1, 3}]"""
        pass

if __name__ == '__main__':
    T = TestLowerStar()
    T.setup()
    T.test_star()
    T.test_lower_star()
    T.teardown()
