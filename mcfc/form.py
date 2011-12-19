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

# UFL libs
from ufl.finiteelement import FiniteElement, VectorElement, TensorElement
# MCFC libs
from codegeneration import *
from formutils import *
from utilities import uniqify, formElementRank, formElementSpaceDim

class FormBackend(object):
    "Base class for generating form tabulation kernels."

    numNodesPerEle = 3
    numDimensions = 2
    numGaussPoints = 6

    def __init__(self):
        self._expressionBuilder = None
        self._quadratureExpressionBuilder = None
        self._coefficientUseFinder = CoefficientUseFinder()

    def compile(self, name, form, outerScope = None):
        "Compile a form with a given name."

        # FIXME what if we have multiple integrals?
        integrand = form.integrals()[0].integrand()
        form_data = form.form_data()
        assert form_data, "Form has no form data attached!"
        rank = form_data.rank

        # Get parameter list for kernel declaration.
        formalParameters, actualParameters = self._buildKernelParameters(integrand, form)
        # Attach list of formal and actual kernel parameters to form data
        form.form_data().formalParameters = formalParameters
        form.form_data().actualParameters = actualParameters

        # Initialise basis tensors if necessary
        declarations = self._buildBasisTensors(form_data)

        # Build the loop nest
        loopNest = self.buildLoopNest(form)
        statements = [loopNest]

        # Initialise the local tensor values to 0
        initialiser = self.buildLocalTensorInitialiser(form)
        depth = rank
        loopBody = getScopeFromNest(loopNest, depth)
        loopBody.prepend(initialiser)

        # Insert the expressions into the loop nest
        partitions = findPartitions(integrand)
        for (tree, depth) in partitions:
            expression = self.buildExpression(form, tree)
            exprDepth = depth + rank + 1 # add 1 for quadrature loop
            loopBody = getScopeFromNest(loopNest, exprDepth)
            loopBody.prepend(expression)

        # If there's any coefficients, we need to build a loop nest
        # that calculates their values at the quadrature points
        # Note: this uses data generated during building of the expressions,
        # hence needs to be done afterwards, though it comes first in the
        # generated code
        if form_data.num_coefficients > 0:
            declarations += self.buildCoeffQuadDeclarations(form)
            statements = [self.buildQuadratureLoopNest(form)] + statements

        # If we are given an outer scope, append the statements to it
        if outerScope:
            for s in statements:
                outerScope.append(s)
            statements = [outerScope]

        # Build the function with the loop nest inside
        body = Scope(declarations + statements)
        kernel = FunctionDefinition(Void(), name, formalParameters, body)

        return kernel

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

    def buildCoefficientLoopNest(self, coeff, rank, scope):
        "Build loop nest evaluating a coefficient at a given quadrature point."

        loop = scope

        # Build loop over the correct number of dimensions
        for r in range(rank):
            indVar = self.buildDimIndex(r).name()
            dimLoop = buildSimpleForLoop(indVar, self.numDimensions)
            loop.append(dimLoop)
            loop = dimLoop

        # Add initialiser here
        initialiser = self.buildCoeffQuadratureInitialiser(coeff)
        loop.append(initialiser)

        # One loop over the basis functions
        indVar = self.buildBasisIndex(0).name()
        basisLoop = buildSimpleForLoop(indVar, self.numNodesPerEle)
        loop.append(basisLoop)

        # Add the expression to compute the value inside the basis loop
        computation = self.buildQuadratureExpression(coeff)
        basisLoop.append(computation)

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

    # This function provides a simple calculation of the number of basis
    # functions per element. This works for the tensor product of a scalar basis
    # only.
    def _numBasisFunctions(self, form):
        elementRank = formElementRank(form)
        spaceDimension = formElementSpaceDim(form)
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

    def _buildKernelParameters(self, tree, form):
        raise NotImplementedError("You're supposed to implement _buildKernelParameters()!")

    def subscript_detwei(self, tree):
        raise NotImplementedError("You're supposed to implement subscript_detwei()!")

# Variables used globally

detwei = Variable("detwei", Pointer(Real()) )
timestep = Variable("dt", Real() )
localTensor = Variable("localTensor", Pointer(Real()) )

# vim:sw=4:ts=4:sts=4:et
