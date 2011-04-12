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


"""form.py - contains the code shared by different form backends, e.g.
cudaform.py, op2form.py, etc."""

# MCFC libs
from codegeneration import *
from symbolicvalue import SymbolicValue
from utilities import uniqify
# UFL libs
import ufl.argument
import ufl.coefficient
from ufl.algorithms.transformations import Transformer

class ExpressionBuilder(Transformer):

    def build(self, tree):
        self._exprStack = []
        # When we pass through the first IndexSum, this will get incremented
        # to 0, which is the count of the first dim index
        self._indexSumDepth = -1 
        self.visit(tree)

        expr = self._exprStack.pop()

        if len(self._exprStack) is not 0:
            raise RuntimeError("Expression stack not empty.")

        # Everything needs to be multiplied by detwei
        indices = self.subscript_detwei()
        detwei = Variable("detwei")
        detweiExpr = self.buildSubscript(detwei, indices)
        expr = MultiplyOp(expr, detweiExpr)

        return expr

    def component_tensor(self, tree, *ops):
        pass

    def indexed(self, tree, *ops):
        pass

    # We need to keep track of how many IndexSums we passed through
    # so that we know which dim index we're dealing with.
    def index_sum(self, tree):
        summand, indices = tree.operands()
        self._indexSumDepth = self._indexSumDepth + 1
        self.visit(summand)
        self._indexSumDepth = self._indexSumDepth - 1

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

        depth = self._indexSumDepth
        indices = self.subscript(tree, depth)
        spatialDerivExpr = self.buildSubscript(base, indices)
        self._exprStack.append(spatialDerivExpr)
 
    def argument(self, tree):
        name = buildArgumentName(tree)
        base = Variable(name)

        indices = self.subscript(tree)
        argExpr = self.buildSubscript(base, indices)
        self._exprStack.append(argExpr)

    def coefficient(self, tree):
        coeffExpr = self.buildCoeffQuadratureAccessor(tree)
        self._exprStack.append(coeffExpr)

    def buildCoeffQuadratureAccessor(self, coeff):
        if isinstance(coeff, ufl.coefficient.Coefficient):
            name = buildCoefficientQuadName(coeff)
        else:
            name = buildSpatialDerivativeName(coeff)
        base = Variable(name)
        
        indices = self.subscript_CoeffQuadrature(coeff)

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

    def subscript_detwei(self, tree):
        raise NotImplementedError("You're supposed to implement subscript_detwei()!")

class QuadratureExpressionBuilder:

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

    

class FormBackend:

    def __init__(self):
        self._expressionBuilder = None
        self._quadratureExpressionBuilder = None
        self._coefficientUseFinder = CoefficientUseFinder()

    def buildExpression(self, form, tree):
        "Build the expression represented by the subtree tree of form."
        # Build the rhs expression
        rhs = self._expressionBuilder.build(tree)

        # Assign expression to the local tensor value
        lhs = self._expressionBuilder.buildLocalTensorAccessor(form)
        expr = PlusAssignmentOp(lhs, rhs)

        return expr
        
    def buildQuadratureExpression(self, coeff):
        rhs = self._quadratureExpressionBuilder.build(coeff)

        lhs = self._expressionBuilder.buildCoeffQuadratureAccessor(coeff)
        expr = PlusAssignmentOp(lhs, rhs)
        
        return expr
    
    def buildLocalTensorInitialiser(self, form):
        lhs = self._expressionBuilder.buildLocalTensorAccessor(form)
        rhs = Literal(0.0)
        initialiser = AssignmentOp(lhs, rhs)
        return initialiser

    def buildCoeffQuadratureInitialiser(self, coeff):
        accessor = self._expressionBuilder.buildCoeffQuadratureAccessor(coeff)
        initialiser = AssignmentOp(accessor, Literal(0.0))
        return initialiser

    def buildCoeffQuadDeclarations(self, form):
        integrand = form.integrals()[0].integrand()
        coefficients, spatialDerivatives = self._coefficientUseFinder.find(integrand)
        
        declarations = []

        for coeff in coefficients:
            name = buildCoefficientQuadName(coeff)
            rank = coeff.rank()
            decl = self._buildCoeffQuadDeclaration(name, rank)
            declarations.append(decl)

        for d in spatialDerivatives:
            name = buildSpatialDerivativeName(d)
            operand = d.operands()[0]
            rank = operand.rank() + 1 # The extra dimension due to the differentiation
            decl = self._buildCoeffQuadDeclaration(name, rank)
            declarations.append(decl)

        return declarations

    def _buildCoeffQuadDeclaration(self, name, rank):
        length = numGaussPoints * pow(numDimensions, rank)
        t = Array(Real(), Literal(length))
        var = Variable(name, t)
        decl = Declaration(var)
        return decl

    def compile(self, form):
        raise NotImplementedError("You're supposed to implement compile()!")

class CoefficientUseFinder(Transformer):
    """Finds the nodes that 'use' a coefficient. This is either a Coefficient
    itself, or a SpatialDerivative that has a Coefficient as its operand"""

    def find(self, tree):
        # We keep coefficients and spatial derivatives in separate lists
        # because we need separate criteria to uniqify the lists.
        self._coefficients = []
        self._spatialDerivatives = []
        
        self.visit(tree)

        # Coefficients define __eq__ and __hash__ so the straight uniqify works.
        # For spatial derivatives, we need to compare the coefficients.
        coefficients = uniqify(self._coefficients)
        spatialDerivatives = uniqify(self._spatialDerivatives, lambda x: x.operands()[0])

        return coefficients, spatialDerivatives

    # Most expressions are uninteresting.
    def expr(self, tree, *ops):
        pass

    def argument(self, tree):
        pass

    def spatial_derivative(self, tree):
        subject = tree.operands()[0]
        if isinstance(subject, ufl.coefficient.Coefficient):
            self._spatialDerivatives.append(tree)

    def coefficient(self, tree):
        self._coefficients.append(tree)

class IndexSumCounter(Transformer):
    "Count how many IndexSums are nested inside a tree."

    def count(self, tree):
        self._indexSumDepth = 0
        self._maxIndexSumDepth = 0
        self.visit(tree)
        return self._maxIndexSumDepth

    def index_sum(self, tree):
        
        summand, indices = tree.operands()

        self._indexSumDepth = self._indexSumDepth + 1
        if self._indexSumDepth > self._maxIndexSumDepth:
            self._maxIndexSumDepth = self._indexSumDepth

        self.visit(summand)

        self._indexSumDepth = self._indexSumDepth - 1

    # We don't care about any other node.
    def expr(self, tree, *ops):
        pass

    def terminal(self, tree):
        pass

class Partitioner(Transformer):
    """Partitions the expression up so that each partition fits inside
    strictly on loop in the local assembly loop nest.
    Returns a list of the partitions, and their depth (starting from
    inside the quadrature loop)."""

    def partition(self, tree):
        self._partitions = []
        self._ISC = IndexSumCounter()
        self.visit(tree)
        return self._partitions

    def sum(self, tree):
        ops = tree.operands()
        lDepth = self._ISC.count(ops[0])
        rDepth = self._ISC.count(ops[1])
        # If both sides have the same nesting level:
        if lDepth == rDepth:
            self._partitions.append((tree,lDepth))
            return
        else:
            self.visit(ops[0])
            self.visit(ops[1])
    
    # If it's not a sum, then there shouldn't be any partitioning
    # of the tree anyway.
    def expr(self, tree):
        depth = self._ISC.count(tree)
        self._partitions.append((tree,depth))

def findPartitions(tree):
    part = Partitioner()
    return part.partition(tree)

# Code indices represent induction variables in a loop. A list of CodeIndexes is
# supplied to buildSubscript, in order for it to build the computation of a
# subscript.

class CodeIndex:

    def __init__(self, count=None):
        self._count = count

class BasisIndex(CodeIndex):
    
    def extent(self):
        return Literal(numNodesPerEle)

    def name(self):
        return basisInductionVariable(self._count)

class GaussIndex(CodeIndex):

    def extent(self):
        return Literal(numGaussPoints)

    def name(self):
        return gaussInductionVariable()

class DimIndex(CodeIndex):

    def extent(self):
        return Literal(numDimensions)

    def name(self):
        return dimInductionVariable(self._count)

# Name builders

def buildArgumentName(tree):
    element = tree.element()
    if element.num_sub_elements() is not 0:
        if element.family() == 'Mixed':
            raise NotImplementedError("I can't digest mixed elements. They make me feel funny.")
        else:
            # Sub elements are all the same
            sub_elements = element.sub_elements()
            element = sub_elements[0]
        
    name = element.shortstr()
    return name

def buildSpatialDerivativeName(tree):
    operand = tree.operands()[0]
    if isinstance(operand, ufl.argument.Argument):
        name = buildArgumentName(operand)
    elif isinstance(operand, ufl.coefficient.Coefficient):
        name = buildCoefficientQuadName(operand)
    else:
        cls = operand.__class__.__name__
        raise NotImplementedError("Unsupported SpatialDerivative of " + cls)
    spatialDerivName = 'd_%s' % (name)
    return spatialDerivName

def buildCoefficientName(tree):
    count = tree.count()
    name = 'c%d' % (count)
    return name

def buildCoefficientQuadName(tree):
    count = tree.count()
    name = 'c_q%d' %(count)
    return name

# Names of induction variables

def gaussInductionVariable():
    return "i_g"

def basisInductionVariable(count):
    name = "i_r_%d" % (count)
    return name

def dimInductionVariable(count):
    name = "i_d_%d" % (count)
    return name

# Global variables for code generation.
# Eventually these need to be set by the caller of the code generator
numNodesPerEle = 3
numDimensions = 2
numGaussPoints = 6

# Variables used globally

detwei = Variable("detwei", Pointer(Real()) )
timestep = Variable("dt", Real() )
localTensor = Variable("localTensor", Pointer(Real()) )

# vim:sw=4:ts=4:sts=4:et
