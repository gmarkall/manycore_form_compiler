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

# UFL modules
import ufl
from ufl.tensoralgebra import Inverse, Determinant
# femtools libs
from get_element import get_element

countJ = -3

# Introduce convenience shorthand for initialising TestFunction from Coefficient
def TestFunction(initialiser):
    "UFL value: Create a test function argument to a form."
    if isinstance(initialiser, ufl.Coefficient):
        initialiser = initialiser.element()
    return ufl.TestFunction(initialiser)

# Introduce convenience shorthand for initialising TrialFunction from Coefficient
def TrialFunction(initialiser):
    "UFL value: Create a trial function argument to a form."
    if isinstance(initialiser, ufl.Coefficient):
        initialiser = initialiser.element()
    return ufl.TrialFunction(initialiser)

def Jacobian(coordinates):
    "UFL value: Create a Jacobian matrix coefficient from the coordinate field."
    # If the coordinate field lives in P^n, the Jacobian lives in P^{n-1}_DG
    degree = coordinates.element().degree() - 1
    domain = coordinates.element().cell().domain()
    # FIXME Evil hack!
    # We attach the coordinate field as the quad_scheme property, since this
    # is not affected by reconstruction
    element = ufl.TensorElement('DG', domain, degree, quad_scheme=coordinates)
    return ufl.Coefficient(element, countJ)

# Number of facets associated with each UFL domain
domain2num_vertices = {"cell1D": None,
                       "cell2D": None,
                       "cell3D": None,
                       "interval": 2,
                       "triangle": 3,
                       "tetrahedron": 4,
                       "quadrilateral": 4,
                       "hexahedron": 8}

def transform(coordinates):
    element = coordinates.element()
    domain = element.cell().domain()
    # Query Femtools for:
    # - quadrature weights
    # - quadrature points
    # - shape functions
    # - shape function derivatives
    element._weight, element._l, element._n, element._dn \
        = get_element( element.cell().topological_dimension(),
                       domain2num_vertices[domain],
                       element.quadrature_scheme(),
                       element.degree() )
    # Set the quadrature scheme of the coordinate element to the
    # number of quadrature points for query by the form backend
    element._quad_scheme = len(element._weight)
    J = Jacobian(coordinates)
    invJ = Inverse(J)
    detJ = Determinant(J)
    return J, invJ, detJ

# vim:sw=4:ts=4:sts=4:et
