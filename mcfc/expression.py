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
from symbolicvalue import SymbolicValue
from ufl.finiteelement import FiniteElement, VectorElement, TensorElement
from ufl.argument import Argument
from ufl.common import Stack
from ufl.indexing import Index

class ExpressionBuilder(Transformer):

    def __init__(self, formBackend):
        Transformer.__init__(self)
        self._formBackend = formBackend

    def build(self, tree):
        "Build the rhs for evaluating an expression tree."
        self._exprStack = []
        self._indexStack = Stack()
        self.visit(tree)

        expr = self._exprStack.pop()

        if len(self._exprStack) is not 0:
            raise RuntimeError("Expression stack not empty.")

        return expr

    def component_tensor(self, tree, *ops):
        pass

    # When entering an index, we need to memorise the indices that are attached 
    # to it before descending into the sub-tree. The sub-tree handlers can make
    # use of these memorised indices in their code generation.
    def indexed(self, tree):
        o, i = tree.operands()
        self._indexStack.push(self.visit(i))
        self.visit(o)
        self._indexStack.pop()

    def multi_index(self, tree):
        indices = []
        for i in tree:
            if isinstance(i, Index):
                indices.append(i)
            else:
                raise RuntimeError('Other types of indices not yet implemented.')
        return tuple(indices)
            

    # We need to keep track of how many IndexSums we passed through
    # so that we know which dim index we're dealing with.
    def index_sum(self, tree):
        summand, mi = tree.operands()

        self.visit(summand)

    def constant_value(self, tree):
        if isinstance(tree, SymbolicValue):
            value = Variable(tree.value())
        else:
            value = Literal(tree.value())
        self._exprStack.append(value)

    def sum(self, tree, *ops):
        rhs = self._exprStack.pop()
        lhs = self._exprStack.pop()
        add = AddOp(lhs, rhs)
        self._exprStack.append(add)

    def product(self, tree, *ops):
        rhs = self._exprStack.pop()
        lhs = self._exprStack.pop()
        multiply = MultiplyOp(lhs, rhs)
        self._exprStack.append(multiply)

    def spatial_derivative(self, tree):
        name = buildSpatialDerivativeName(tree)
        base = Variable(name)

        dimIndices = self._indexStack.peek()
        indices = self.subscript(tree, dimIndices)
        spatialDerivExpr = self.buildSubscript(base, indices)
        self._exprStack.append(spatialDerivExpr)
 
    def argument(self, tree):
        e = tree.element()
        indices = self.subscript(tree)
        
        if isinstance(e, FiniteElement):
            base = Variable(buildArgumentName(tree))
            argExpr = self.buildSubscript(base, indices)
        elif isinstance(e, VectorElement):
            base = Variable(buildVectorArgumentName(tree))
            argExpr = self.buildMultiArraySubscript(base, indices)
        else:
            base = Variable(buildTensorArgumentName(tree))
            argExpr = self.buildMultiArraySubscript(base, indices)

        self._exprStack.append(argExpr)

    def coefficient(self, tree):
        coeffExpr = self.buildCoeffQuadratureAccessor(tree)
        self._exprStack.append(coeffExpr)

    def buildCoeffQuadratureAccessor(self, coeff, fake_indices=False):
        rank = coeff.rank()
        if isinstance(coeff, ufl.coefficient.Coefficient):
            name = buildCoefficientQuadName(coeff)
        else:
            # The spatial derivative adds an extra dim index so we need to
            # bump up the rank
            rank = rank + 1
            name = buildSpatialDerivativeName(coeff)
        base = Variable(name)
        
        # If there are no indices present (e.g. in the quadrature evaluation loop) then
        # we need to fake indices for the coefficient based on its rank:
        if fake_indices:
            fake = []
            for i in range(rank):
                fake.append(Index(i))
            self._indexStack.push(tuple(fake))

        indices = self.subscript_CoeffQuadrature(coeff)

        # Remove the fake indices
        if fake_indices:
            self._indexStack.pop()

        coeffExpr = self.buildSubscript(base, indices)
        return coeffExpr

    def buildLocalTensorAccessor(self, form):
        indices = self.subscript_LocalTensor(form)
        
        # Subscript the local tensor variable
        expr = self.buildSubscript(localTensor, indices)
        return expr

    def buildSubscript(self, variable, indices):
        raise NotImplementedError("You're supposed to implement buildSubscript()!")

    def subscript_LocalTensor(self, form):
        raise NotImplementedError("You're supposed to implement subscript_LocalTensor()!")

    def subscript_CoeffQuadrature(self, coeff):
        raise NotImplementedError("You're supposed to implement subscript_CoeffQuadrature()!")

    def subscript(self, tree):
        raise NotImplementedError("You're supposed to implement subscript()!")

class QuadratureExpressionBuilder:

    def __init__(self, formBackend):
        self._formBackend = formBackend

    def build(self, tree):
        # Build Accessor for values at nodes
        indices = self.subscript(tree)
        
        if isinstance(tree, ufl.coefficient.Coefficient):
            name = buildCoefficientName(tree)
        elif isinstance (tree, ufl.differentiation.SpatialDerivative):
            operand, _ = tree.operands()
            name = buildCoefficientName(operand)

        coeffAtBasis = Variable(name)
        coeffExpr = self.buildSubscript(coeffAtBasis, indices)
        
        # Build accessor for argument
        if isinstance(tree, ufl.coefficient.Coefficient):
            name = buildArgumentName(tree)
            indices = self.subscript_argument(tree)
        elif isinstance (tree, ufl.differentiation.SpatialDerivative):
            operand, indices = tree.operands()
            element = operand.element()
            basis = ufl.argument.TrialFunction(element)
            basisDerivative = ufl.differentiation.SpatialDerivative(basis, indices)
            name = buildSpatialDerivativeName(basisDerivative)
            indices = self.subscript_spatial_derivative(basisDerivative)

        arg = Variable(name)
        argExpr = self.buildSubscript(arg, indices)

        # Combine to form the expression
        expr = MultiplyOp(coeffExpr, argExpr)
        return expr

    def buildSubscript(self, variable, indices):
        raise NotImplementedError("You're supposed to implement buildSubscript()!")

    def subscript(self, tree):
        raise NotImplementedError("You're supposed to implement subscript()!")

    def subscript_argument(self, tree):
        raise NotImplementedError("You're supposed to implement subscript_argument()!")

    def subscript_spatial_derivative(self, tree):
        raise NotImplementedError("You're supposed to implement subscript_spatial_derivative()!")

# vim:sw=4:ts=4:sts=4:et
