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

from ufl import TensorElement, FiniteElement
from uflnamespace import Jacobian, JacobianInverse, JacobianDeterminant

class Transformation:

    def __init__(self, coordinates):
        # If the coordinate field lives in P^n, the Jacobian lives in P^{n-1}_DG
        degree = coordinates.element().degree() - 1
        # FIXME: hardcoded for triangles
        self.J = Jacobian(TensorElement('DG', 'triangle', degree))
        self.invJ = JacobianInverse(TensorElement('DG', 'triangle', degree))
        self.detJ = JacobianDeterminant(FiniteElement('DG', 'triangle', degree))

def transform(coordinates):
    return Transformation(coordinates)

# vim:sw=4:ts=4:sts=4:et
