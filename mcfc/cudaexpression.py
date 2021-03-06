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

# Variables

threadCount = Variable("THREAD_COUNT")
threadId = Variable("THREAD_ID")
numElements = Variable("n_ele", Integer() )

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

# Expression builders

class CudaExpressionBuilder(ExpressionBuilder):

    def buildSubscript(self, variable, indices):
        return buildSubscript(variable, indices)

    def subscript_LocalTensor(self, form_data):
        indices = super(CudaExpressionBuilder,self).subscript_LocalTensor(form_data)
        # First index is the element index
        return [ElementIndex()] + indices

    def symbolic_value(self, value):
        return Variable(value)

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
        indices.append(buildQuadratureBasisIndex(0, extract_subelement(tree)))
        return indices

# vim:sw=4:ts=4:sts=4:et
