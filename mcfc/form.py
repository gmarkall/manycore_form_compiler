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
from utilities import uniqify
# UFL libs
from ufl.argument import Argument
from ufl.coefficient import Coefficient
from ufl.algorithms.transformations import Transformer
from ufl.finiteelement import FiniteElement, VectorElement, TensorElement

class FormBackend(object):
    "Base class for generating form tabulation kernels."

    numNodesPerEle = 3
    numDimensions = 2
    numGaussPoints = 6

    def __init__(self):
        self._expressionBuilder = None
        self._quadratureExpressionBuilder = None
        self._coefficientUseFinder = CoefficientUseFinder()

    def buildBasisIndex(self, count):
        "Build index for a loop over basis function values."
        return BasisIndex(self.numNodesPerEle, count)

    def buildDimIndex(self, count):
        "Build index for a loop over spatial dimensions."
        return DimIndex(self.numDimensions, count)

    def buildConstDimIndex(self, count):
        "Build literal subscript for a loop over spatial dimensions."
        return ConstIndex(self.numDimensions, count)

    def buildGaussIndex(self):
        "Build index for a Gauss quadrature loop."
        return GaussIndex(self.numGaussPoints)

    def buildExpression(self, form, tree):
        "Build the expression represented by the subtree tree of form."
        # Build the rhs expression
        rhs = self._expressionBuilder.build(tree)

        # The rhs of a form needs to be multiplied by detwei
        indices = self.subscript_detwei()
        detwei = Variable("detwei")
        detweiExpr = self._expressionBuilder.buildSubscript(detwei, indices)
        rhs = MultiplyOp(rhs, detweiExpr)

        # Assign expression to the local tensor value
        lhs = self._expressionBuilder.buildLocalTensorAccessor(form)
        expr = PlusAssignmentOp(lhs, rhs)

        return expr

    def buildLoopNest(self, form):
        "Build the loop nest for evaluating a form expression."
        rank = form.form_data().rank
        numBasisFunctions = self._numBasisFunctions(form)

        # FIXME what if we have multiple integrals?
        integrand = form.integrals()[0].integrand()

        # Build the loop over the first rank, which always exists
        indVarName = self.buildBasisIndex(0).name()
        loop = buildSimpleForLoop(indVarName, numBasisFunctions)
        outerLoop = loop

        # Add another loop for each rank of the form (probably no
        # more than one more... )
        for r in range(1,rank):
            indVarName = self.buildBasisIndex(r).name()
            basisLoop = buildSimpleForLoop(indVarName, numBasisFunctions)
            loop.append(basisLoop)
            loop = basisLoop

        # Add a loop for the quadrature
        indVarName = self.buildGaussIndex().name()
        gaussLoop = buildSimpleForLoop(indVarName, self.numGaussPoints)
        loop.append(gaussLoop)
        loop = gaussLoop

        # Determine how many dimension loops we need by inspection.
        # We count the nesting depth of IndexSums to determine
        # how many dimension loops we need.
        dimLoops = indexSumIndices(integrand)

        # Add loops for each dimension as necessary.
        for d in dimLoops:
            indVarName = self.buildDimIndex(d['count']).name()
            dimLoop = buildSimpleForLoop(indVarName, d['extent'])
            loop.append(dimLoop)
            loop = dimLoop

        # Hand back the outer loop, so it can be inserted into some
        # scope.
        return outerLoop

    def buildQuadratureExpression(self, coeff):
        "Build the expression to evaluate a particular coefficient."
        rhs = self._quadratureExpressionBuilder.build(coeff)

        lhs = self._expressionBuilder.buildCoeffQuadratureAccessor(coeff, True)
        expr = PlusAssignmentOp(lhs, rhs)

        return expr

    def buildQuadratureLoopNest(self, form):
        "Build quadrature loop nest evaluating all coefficients of the form."

        # FIXME what if we have multiple integrals?
        integrand = form.integrals()[0].integrand()
        coefficients, spatialDerivatives = self._coefficientUseFinder.find(integrand)

        # Outer loop over gauss points
        indVar = self.buildGaussIndex().name()
        gaussLoop = buildSimpleForLoop(indVar, self.numGaussPoints)

        # Build a loop nest for each coefficient containing expressions
        # to compute its value
        for coeff in coefficients:
            rank = coeff.rank()
            self.buildCoefficientLoopNest(coeff, rank, gaussLoop)

        for spatialDerivative in spatialDerivatives:
            operand = spatialDerivative.operands()[0]
            rank = operand.rank() + 1
            self.buildCoefficientLoopNest(spatialDerivative, rank, gaussLoop)

        return gaussLoop

    def buildLocalTensorInitialiser(self, form):
        lhs = self._expressionBuilder.buildLocalTensorAccessor(form)
        rhs = Literal(0.0)
        initialiser = AssignmentOp(lhs, rhs)
        return initialiser

    def buildCoeffQuadratureInitialiser(self, coeff):
        accessor = self._expressionBuilder.buildCoeffQuadratureAccessor(coeff, True)
        initialiser = AssignmentOp(accessor, Literal(0.0))
        return initialiser

    def buildCoeffQuadDeclarations(self, form):
        # FIXME what if we have multiple integrals?
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

    def _elementRank(self, form):
        # Use the element from the first argument, which should be the TestFunction
        arg = form.form_data().arguments[0]
        e = arg.element()

        if isinstance(e, FiniteElement):
            return 0
        elif isinstance(e, VectorElement):
            return 1
        elif isinstance(e, TensorElement):
            return 2
        else:
            raise RuntimeError("Not a recognised element.")

    def _elementSpaceDim(self, form):
        # Use the element from the first argument, which should be the TestFunction
        arg = form.form_data().arguments[0]
        e = arg.element()
        return e.cell().geometric_dimension()

    # This function provides a simple calculation of the number of basis
    # functions per element. This works for the tensor product of a scalar basis
    # only.
    def _numBasisFunctions(self, form):
        elementRank = self._elementRank(form)
        spaceDimension = self._elementSpaceDim(form)
        return self.numNodesPerEle * pow(spaceDimension, elementRank)

    def _buildBasisTensors(self, form_data):
        """When using a basis that is a tensor product of the scalar basis, we
        need to create an array that holds the tensor product. This function
        generates the code to declare and initialise that array."""
        gp = self.numGaussPoints
        nn = self.numNodesPerEle
        nd = self.numDimensions
        initialisers = []

        for a in form_data.actualParameters['arguments']:
            e = a.element()
            # Ignore scalars
            if isinstance(e, FiniteElement):
                continue
            elif isinstance(e, VectorElement):
                n = buildVectorArgumentName(a)
            else:
                raise RuntimeError("Not supported.")

            arg = buildArgumentName(a)
            t = Array(Real(), [nd,gp,nn*nd])
            var = Variable(n, t)

            # Construct the initialiser lists for the tensor product of the scalar basis.
            outer = []
            for d1 in range(nd):
                middle = []
                for igp in range(gp):
                    innermost = []
                    for d2 in range(nd):
                        for inn in range(nn):
                            if d1 == d2:
                                expr = Subscript(Variable(arg), Literal(inn*gp+igp))
                            else:
                                expr = Literal(0.0)
                            innermost.append(expr)
                    middle.append(InitialiserList(innermost))
                outer.append(InitialiserList(middle))
            initlist = InitialiserList(outer)

            init = InitialisationOp(var, initlist)
            initialisers.append(init)

        return initialisers

    def compile(self, form):
        raise NotImplementedError("You're supposed to implement compile()!")

    def _buildKernelParameters(self, tree, form):
        raise NotImplementedError("You're supposed to implement _buildKernelParameters()!")

    def subscript_detwei(self, tree):
        raise NotImplementedError("You're supposed to implement subscript_detwei()!")

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
        if isinstance(subject, Coefficient):
            self._spatialDerivatives.append(tree)

    def coefficient(self, tree):
        self._coefficients.append(tree)

class IndexSumIndexFinder(Transformer):
    "Find the count and extent of indices reduced by an IndexSum in a form."

    def find(self, tree):
        self._indices = []
        self.visit(tree)
        return self._indices

    def index_sum(self, tree):

        summand, mi = tree.operands()
        indices = mi.index_dimensions()

        for c, d in indices.items():
            self._indices.append({"count": c.count(), "extent": d})

        self.visit(summand)

    # We don't care about any other node.
    def expr(self, tree, *ops):
        pass

def indexSumIndices(tree):
    ISIF = IndexSumIndexFinder()
    return ISIF.find(tree)

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

    def __init__(self, extent, count=None):
        self._extent = extent
        self._count = count

    def extent(self):
        return Literal(self._extent)

class BasisIndex(CodeIndex):

    def name(self):
        return "i_r_%d" % (self._count)

class GaussIndex(CodeIndex):

    def name(self):
        return "i_g"

class DimIndex(CodeIndex):

    def name(self):
        return "i_d_%d" % (self._count)

class ConstIndex(CodeIndex):

    def name(self):
        return str(self._count)

# Name builders

def safe_shortstr(name):
    return name[:name.find('(')]

def buildArgumentName(tree):
    element = tree.element()
    if element.num_sub_elements() is not 0:
        if element.family() == 'Mixed':
            raise NotImplementedError("I can't digest mixed elements. They make me feel funny.")
        else:
            # Sub elements are all the same
            sub_elements = element.sub_elements()
            element = sub_elements[0]

    return safe_shortstr(element.shortstr())

def buildSpatialDerivativeName(tree):
    operand = tree.operands()[0]
    if isinstance(operand, Argument):
        name = buildArgumentName(operand)
    elif isinstance(operand, Coefficient):
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

def buildVectorArgumentName(tree):
    return buildArgumentName(tree) + "_v"

def buildVectorSpatialDerivativeName(tree):
    return buildSpatialDerivativeName(tree) + "_v"

def buildTensorArgumentName(tree):
    return buildArgumentName(tree) + "_t"

def buildTensorSpatialDerivativeName(tree):
    return buildSpatialDerivativeName(tree) + "_t"

# Variables used globally

detwei = Variable("detwei", Pointer(Real()) )
timestep = Variable("dt", Real() )
localTensor = Variable("localTensor", Pointer(Real()) )

# vim:sw=4:ts=4:sts=4:et
