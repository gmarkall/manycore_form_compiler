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

from cudaparameters import numElements
from expression import *
from formutils import extract_element

# Variables

threadCount = Variable("THREAD_COUNT")
threadId = Variable("THREAD_ID")

# The ElementIndex is here and not form.py because not all backends need
# an element index (e.g. OP2).

class ElementIndex:

    def extent(self):
        return numElements

    def name(self):
        return "i_ele"

def buildElementLoop():
    indVarName = ElementIndex().name()
    var = Variable(indVarName, Integer())
    init = InitialisationOp(var, threadId)
    test = LessThanOp(var, numElements)
    inc = PlusAssignmentOp(var, threadCount)
    return ForLoop(init, test, inc)

def buildSubscript(variable, indices):
    """Given a list of indices, return an AST that computes
    the offset into the given array using those indices. The order is
    important."""

    # Compute the correct subscript using the Horner scheme, starting with
    # the slowest varying index (last in the list)
    expr = Variable(indices[-1].name())

    # Iteratively build the expression by multiplying the the existing
    # expression with the extent of the next faster varying index and
    # adding the index' iteration variable
    for i in reversed(indices[:-1]):
        expr = MultiplyOp(i.extent(), Bracketed(expr))
        expr = AddOp(Variable(i.name()), expr)

    return Subscript(variable, expr)

def buildMultiArraySubscript(variable, indices):
    """Given a list of indices, return an AST of the variable
    subscripted by the indices as if it were a multidimensional
    array."""

    for i in indices:
        variable = Subscript(variable, Variable(i.name()))

    return variable

# Expression builders

class CudaExpressionBuilder(ExpressionBuilder):

    def buildSubscript(self, variable, indices):
        return buildSubscript(variable, indices)

    def buildMultiArraySubscript(self, variable, indices):
        return buildMultiArraySubscript(variable, indices)

    def subscript_Argument(self, tree):
        # Build the subscript based on the argument count
        count = tree.count()
        element = tree.element()
        indices = []
        for dimIndices in self._indexStack:
            indices.extend(dimIndices)
        indices += [buildBasisIndex(count, element),
                    buildGaussIndex(self._formBackend.numGaussPoints)]
        return indices

    def subscript_SpatialDerivative(self,tree,dimIndices):
        # Build the subscript based on the argument count and the
        # indices
        operand, _ = tree.operands()
        element = operand.element()
        count = operand.count()

        if isinstance(operand, Argument):
            indices = [ ElementIndex()]
            indices.extend(dimIndices)
            indices = indices + [ buildGaussIndex(self._formBackend.numGaussPoints),
                                  buildBasisIndex(count, element) ]
        elif isinstance(operand, Coefficient):
            indices = [ buildGaussIndex(self._formBackend.numGaussPoints) ]
            indices.extend(dimIndices)

        return indices

    def subscript_LocalTensor(self, form):
        form_data = form.form_data()
        rank = form_data.rank

        # First index is the element index
        indices = [ElementIndex()]

        # One rank index for each rank
        for r in range(rank):
            indices.append(buildBasisIndex(r,form))

        return indices

class CudaQuadratureExpressionBuilder(QuadratureExpressionBuilder):

    def buildSubscript(self, variable, indices):
        return buildSubscript(variable, indices)

    def subscript(self, tree):
        # We index vector/tensor valued coefficients by separate indices for
        # the scalar basis and the spatial dimension(s). Hence we need to
        # extract the scalar element to build the appropriate basis index.
        indices = [ ElementIndex() ]
        for r in range(tree.rank()):
            indices.append(buildDimIndex(r,tree))
        indices.append(buildBasisIndex(0, extract_subelement(tree)))
        return indices

    def subscript_spatial_derivative(self, tree):
        # The count of the basis function induction variable is always
        # 0 in the quadrature loops (i.e. i_r_0), and only the first dim
        # index should be used to subscript the derivative (I think).
        indices = [ ElementIndex(),
                    buildDimIndex(0, tree),
                    buildGaussIndex(self._formBackend.numGaussPoints),
                    buildBasisIndex(0, extract_subelement(tree)) ]
        return indices

# vim:sw=4:ts=4:sts=4:et
