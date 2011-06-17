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

from form import *
from cudaparameters import numElements

# The ElementIndex is here and not form.py because not all backends need
# an element index (e.g. OP2).

class ElementIndex(CodeIndex):

    def extent(self):
        return numElements

    def name(self):
        return eleInductionVariable()

def eleInductionVariable():
    return "i_ele"

def buildSubscript(variable, indices):
    """Given a list of indices, return an AST that computes
    the offset into the given array using those indices. The order is
    important."""

    # Start our expression with the first index
    name = indices[0].name()
    offset = Variable(name)
    
    # Compute the expression for all indices
    for v in range(1,len(indices)):
        subindices = indices[:v]
        name = indices[v].name()
        expr = Variable(name)
        
        # Find the correct offset for this index
        for u in range(len(subindices)):
            multiplier = subindices[u].extent()
            expr = MultiplyOp(multiplier, expr)
        offset = AddOp(offset, expr)
    
    return Subscript(variable, offset)

# Expression builders

class CudaExpressionBuilder(ExpressionBuilder):

    def buildSubscript(self, variable, indices):
        return buildSubscript(variable, indices)

    def subscript(self, tree, depth=None):
        meth = getattr(self, "subscript_"+tree.__class__.__name__)
        if depth is None:
            return meth(tree)
        else:
            return meth(tree, depth)

    def subscript_Argument(self, tree):
        # Build the subscript based on the argument count
        count = tree.count()
        indices = [BasisIndex(count), GaussIndex()]
        return indices

    def subscript_SpatialDerivative(self,tree,depth):
        # Build the subscript based on the argument count and the
        # nesting depth of IndexSums of the expression.
        operand, _ = tree.operands()
        count = operand.count()

        if isinstance(operand, ufl.argument.Argument):
            indices = [ElementIndex(), DimIndex(depth), GaussIndex(), BasisIndex(count)]
        elif isinstance(operand, ufl.coefficient.Coefficient):
            indices = [ GaussIndex() ]
            depth = operand.rank() + 1

            for r in range(depth):
                indices.append(DimIndex(r))

        return indices

    def subscript_detwei(self):
        indices = [ElementIndex(), GaussIndex()]
        return indices

    def subscript_LocalTensor(self, form):
        form_data = form.form_data()
        rank = form_data.rank
        
        # First index is the element index
        indices = [ElementIndex()]

        # One rank index for each rank
        for r in range(rank):
            indices.append(BasisIndex(r))

        return indices

    def subscript_CoeffQuadrature(self, coeff):
        # Build the subscript based on the rank
        indices = [GaussIndex()]
        depth = coeff.rank()
        if isinstance(coeff, ufl.differentiation.SpatialDerivative):
            # We need to add one, since the differentiation added a 
            # dim index
            depth = depth + 1
        for r in range(depth):
            indices.append(DimIndex(r))
        
        return indices

class CudaQuadratureExpressionBuilder(QuadratureExpressionBuilder):

    def buildSubscript(self, variable, indices):
        return buildSubscript(variable, indices)

    def subscript(self, tree):
        rank = tree.rank()
        indices = [ ElementIndex() ]
        for r in range(rank):
            index = DimIndex(r)
            indices.append(index)
        indices.append(BasisIndex(0))
        return indices

    def subscript_argument(self, tree):
        # The count of the basis function induction variable is always
        # 0 in the quadrature loops (i.e. i_r_0)
        indices = [BasisIndex(0), GaussIndex()]
        return indices

    def subscript_spatial_derivative(self, tree):
        # The count of the basis function induction variable is always
        # 0 in the quadrature loops (i.e. i_r_0), and only the first dim
        # index should be used to subscript the derivative (I think).
        argument = tree.operands()[0]
        count = argument.count()
        indices = [ElementIndex(), DimIndex(0), GaussIndex(), BasisIndex(0)]
        return indices
