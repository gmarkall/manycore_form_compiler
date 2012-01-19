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
from ufl.coefficient import Coefficient
# MCFC libs
from codegeneration import *
from formutils import *
from utilities import uniqify

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
        loopNest = self.buildExpressionLoopNest(form)
        statements = [loopNest]

        # Initialise the local tensor values to 0
        initialiser = self.buildLocalTensorInitialiser(form)
        depth = rank
        loopBody = getScopeFromNest(loopNest, depth)
        loopBody.prepend(initialiser)

        # Insert the expressions into the loop nest
        partitions = findPartitions(integrand)
        for (tree, depth) in partitions:
            expression, subexpressions = self.buildExpression(form, tree)
            exprDepth = depth + rank + 1 # add 1 for quadrature loop
            loopBody = getScopeFromNest(loopNest, exprDepth)
            loopBody.prepend(expression)
            for expr in subexpressions:
                getScopeFromNest(loopNest, rank + 1).prepend(expr)

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

    def buildExpression(self, form, tree):
        "Build the expression represented by the subtree tree of form."
        # Build the rhs expression
        rhs, subexpr = self._expressionBuilder.build(tree)

        # The rhs of a form needs to be multiplied by detwei
        indices = self.subscript_detwei()
        detwei = Variable("detwei")
        detweiExpr = self._expressionBuilder.buildSubscript(detwei, indices)
        rhs = MultiplyOp(rhs, detweiExpr)

        # Assign expression to the local tensor value
        lhs = self._expressionBuilder.buildLocalTensorAccessor(form)
        expr = PlusAssignmentOp(lhs, rhs)

        return expr, subexpr

    def buildExpressionLoopNest(self, form):
        "Build the loop nest for evaluating a form expression."

        # FIXME what if we have multiple integrals?
        integrand = form.integrals()[0].integrand()

        # Build the loop over the first rank, which always exists
        loop = buildIndexForLoop(buildBasisIndex(0, form))
        outerLoop = loop

        # Add a loop over basis functions for each rank of the form
        for r in range(1,form.form_data().rank):
            basisLoop = buildIndexForLoop(buildBasisIndex(r, form))
            loop.append(basisLoop)
            loop = basisLoop

        # Add a loop for the quadrature
        gaussLoop = buildIndexForLoop(buildGaussIndex(self.numGaussPoints))
        loop.append(gaussLoop)
        loop = gaussLoop

        # Find the indices over dimensions and and loops for them.
        for index in indexSumIndices(integrand):
            dimLoop = buildIndexForLoop(index)
            loop.append(dimLoop)
            loop = dimLoop

        # Hand back the outer loop, so it can be inserted into some
        # scope.
        return outerLoop

    def buildCoefficientLoopNest(self, coeff, rank, loop):
        "Build loop nest evaluating a coefficient at a given quadrature point."

        # Build loop over the correct number of dimensions
        for r in range(rank):
            dimLoop = buildIndexForLoop(buildDimIndex(r, coeff))
            loop.append(dimLoop)
            loop = dimLoop

        # Add initialiser here
        loop.append(self.buildCoeffQuadratureInitialiser(coeff))

        # One loop over the basis functions
        # FIXME: We fall back to the basis of the scalar element
        basisLoop = buildIndexForLoop(buildBasisIndex(0, extract_subelement(coeff)))
        loop.append(basisLoop)

        # Add the expression to compute the value inside the basis loop
        basisLoop.append(self.buildQuadratureExpression(coeff))

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
        gaussLoop = buildIndexForLoop(buildGaussIndex(self.numGaussPoints))

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
