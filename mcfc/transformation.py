# This file is part of the Manycore Form Compiler.
#
# The Manycore Form Compiler is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# The Manycore Form Compiler is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# the Manycore Form Compiler.  If not, see <http://www.gnu.org/licenses>
#
# Copyright (c) 2011, Graham Markall <grm08@doc.ic.ac.uk> and others. Please see
# the AUTHORS file in the main source directory for a full list of copyright
# holders.

# femtools libs
from get_element import get_element
# UFL libs
from ufl import TensorElement, FiniteElement, Coefficient

# Number of facets associated with each UFL domain
domain2num_vertices = {"cell1D": None,
                     "cell2D": None,
                     "cell3D": None,
                     "interval": 2,
                     "triangle": 3,
                     "tetrahedron": 4,
                     "quadrilateral": 4,
                     "hexahedron": 8}

def Jacobian(element):
    """UFL value: Create a Jacobian matrix argument to a form."""
    return Coefficient(element, -3)

def JacobianInverse(element):
    """UFL value: Create a Jacobian inverse argument to a form."""
    return Coefficient(element, -4)

def JacobianDeterminant(element):
    """UFL value: Create a Jacobian determinant argument to a form."""
    return Coefficient(element, -5)

class Transformation:

    # This is _not_ the constructor, since this class is used as a singleton
    def init(self, coordinates):
        e = coordinates.element()
        # Query Femtools for:
        # - quadrature weights
        # - quadrature points
        # - shape functions
        # - shape function derivatives
        e._weight, e._l, e._n, e._dn \
            = get_element( e.cell()._topological_dimension,
                           domain2num_vertices[e.cell().domain()],
                           e.quadrature_scheme(),
                           e.degree() )
        # If the coordinate field lives in P^n, the Jacobian lives in P^{n-1}_DG
        degree = e.degree() - 1
        # FIXME: hardcoded for triangles
        self.J = Jacobian(TensorElement('DG', 'triangle', degree))
        self.invJ = JacobianInverse(TensorElement('DG', 'triangle', degree))
        self.detJ = JacobianDeterminant(FiniteElement('DG', 'triangle', degree))

# Singleton instance of Transformation class
T = Transformation()

# vim:sw=4:ts=4:sts=4:et
