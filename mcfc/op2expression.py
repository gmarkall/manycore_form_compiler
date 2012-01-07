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

from expression import *

def buildSubscript(variable, indices):
    """Given a list of indices, return an AST that subscripts
    the given array using those indices. The order is
    important."""

    # Add subscripts for all indices
    for i in indices:
        variable = Subscript(variable, Variable(i.name()))

    return variable

# Expression builders

class Op2ExpressionBuilder(ExpressionBuilder):

    def buildSubscript(self, variable, indices):
        return buildSubscript(variable, indices)

    def buildMultiArraySubscript(self, variable, indices):
        return buildSubscript(variable, indices)

    def subscript_Argument(self, tree):
        # Build the subscript based on the argument count
        count = tree.count()
        indices = [self._formBackend.buildBasisIndex(count),
                   self._formBackend.buildGaussIndex()]
        return indices

    def subscript_SpatialDerivative(self,tree,dimIndices):
        # Build the subscript based on the argument count and the
        # nesting depth of IndexSums of the expression.
        operand, _ = tree.operands()
        count = operand.count()

        if isinstance(operand, Argument):
            indices = []
            indices.extend(dimIndices)
            indices = indices + [ self._formBackend.buildGaussIndex(),
                                  self._formBackend.buildBasisIndex(count) ]
        elif isinstance(operand, Coefficient):
            indices = [ self._formBackend.buildGaussIndex() ]
            indices.extend(dimIndices)

        return indices

    def subscript_LocalTensor(self, form):
        form_data = form.form_data()
        rank = form_data.rank

        indices = []
        # One rank index for each rank
        for r in range(rank):
            indices.append(self._formBackend.buildBasisIndex(r))

        return indices

class Op2QuadratureExpressionBuilder(QuadratureExpressionBuilder):

    def buildSubscript(self, variable, indices):
        return buildSubscript(variable, indices)

    def subscript(self, tree):
        rank = tree.rank()
        # Subscript order: basis index followed by dimension indices (if any)
        indices = [self._formBackend.buildBasisIndex(0)]
        for r in range(rank):
            indices.append(self._formBackend.buildDimIndex(r))
        return indices

    def subscript_spatial_derivative(self, tree):
        # The count of the basis function induction variable is always
        # 0 in the quadrature loops (i.e. i_r_0), and only the first dim
        # index should be used to subscript the derivative (I think).
        indices = [ self._formBackend.buildDimIndex(0),
                    self._formBackend.buildGaussIndex(),
                    self._formBackend.buildBasisIndex(0) ]
        return indices

# vim:sw=4:ts=4:sts=4:et
