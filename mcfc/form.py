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

# NumPy
from numpy import zeros
# UFL libs
from ufl.differentiation import SpatialDerivative
from ufl.finiteelement import FiniteElement, VectorElement, TensorElement
from ufl.coefficient import Coefficient
# MCFC libs
from codegeneration import *
from formutils import *
from uflnamespace import domain2num_vertices as d2v
from utilities import uniqify

class FormBackend(object):
    "Base class for generating form tabulation kernels."

    def __init__(self):
        self._expressionBuilder = None
        self._quadratureExpressionBuilder = None
        self._coefficientUseFinder = CoefficientUseFinder()

    def compile(self, form):
        "Compile a pre-processed form."

        form_data = form.form_data()
        assert form_data, "Form has no form data attached!"
        self._form_data = form_data

        return GlobalScope([self._compile_integral(i) for i in form_data.named_integrals])
    
    def _compile_integral(self, named_integral, outerScope=None):
        
        integral, name = named_integral
        # NOTE: measure is not yet used but will be necessary when we support
        # surface integrals
        integrand = integral.integrand()
        measure = integral.measure()
        fd = self._form_data
        rank = fd.rank
        
        # Set basic properties of the element
        # FIXME: We only look at the element of the coordinate field for now
        self.numNodesPerEle = d2v[fd.coordinates.cell().domain()]
        self.numDimensions = fd.coordinates.cell().topological_dimension()
        self.numGaussPoints = fd.coordinates.element().quadrature_scheme()

        # Initialise basis tensors if necessary
        declarations = self._buildBasisTensors()

        # Build the loop nest
        loopNest, gaussBody = self.buildExpressionLoopNest(integrand)
        statements = [loopNest]

        # Insert the expressions into the loop nest
        partitions = findPartitions(integrand)
        loopBody = gaussBody
        for (tree, indices) in partitions:
            expression, listexpressions = self.buildExpression(tree)
            buildLoopNest(loopBody, indices).prepend(expression)
            for expr in listexpressions:
                loopBody.prepend(expr)

        # Declare temporary variables to hold subexpression values
        for (tree, _) in partitions:
            initialiser = self.buildSubExprInitialiser(tree)
            loopBody.prepend(initialiser)
        
        # Insert the local tensor expression into the loop nest. There should
        # be no listexpressions.
        expression, _ = self.buildLocalTensorExpression(integrand)
        loopBody.append(expression)

        # If there's any coefficients, we need to build a loop nest
        # that calculates their values at the quadrature points
        # Note: this uses data generated during building of the expressions,
        # hence needs to be done afterwards, though it comes first in the
        # generated code
        if fd.num_coefficients > 0:
            declarations += self.buildCoeffQuadDeclarations(integrand)
            statements = [self.buildQuadratureLoopNest(integrand)] + statements

        # If we are given an outer scope, append the statements to it
        if outerScope:
            for s in statements:
                outerScope.append(s)
            statements = [outerScope]

        # Get parameter list for kernel declaration.
        formalParameters = self._buildKernelParameters(integrand)

        # Build the function with the loop nest inside
        body = Scope(declarations + statements)
        kernel = FunctionDefinition(Void(), name, formalParameters, body)

        return kernel

    def buildLocalTensorExpression(self, tree):
        rhs, listexpr = self._expressionBuilder.build(tree)
        
        # The rhs of an integrand needs to be multiplied by the quadrature weights
        indices = self.subscript_weights()
        weightsExpr = self._expressionBuilder.buildSubscript(weights, indices)
        rhs = MultiplyOp(rhs, weightsExpr)
        
        # Assign expression to the local tensor
        lhs = self._expressionBuilder.buildLocalTensorAccessor(self._form_data)

        expr = PlusAssignmentOp(lhs, rhs)
        return expr, listexpr

    def buildExpression(self, tree):
        "Build the expression represented by the SubTree tree."
        # The tree is rooted by a SubExpr, so we need to construct the 
        # expression beneath it.
        expr = tree.operands()[0]
        rhs, listexpr = self._expressionBuilder.build(expr)

        # Assign expression to the correct Subexpression variable
        lhs = self._expressionBuilder.sub_expr(tree)
        
        expr = PlusAssignmentOp(lhs, rhs)
        return expr, listexpr

    def buildExpressionLoopNest(self, integrand):
        "Build the loop nest for evaluating an integrand expression."

        # Build a loop for the quadrature
        gaussLoop = buildIndexForLoop(buildGaussIndex(self.numGaussPoints))
        
        # Loops over the indices of the local tensor
        outerLoop = self.buildLocalTensorLoops(integrand, gaussLoop)
        
        # Hand back the outer loop, so it can be inserted into some
        # scope.
        return outerLoop, gaussLoop.body()

    def buildLocalTensorLoops(self, integrand, gaussLoop):
        # Build the loop over the first rank, which always exists
        loop = buildIndexForLoop(buildBasisIndex(0, self._form_data))
        outerLoop = loop

        # Add a loop over basis functions for each rank of the form
        for r in range(1, self._form_data.rank):
            basisLoop = buildIndexForLoop(buildBasisIndex(r, self._form_data))
            loop.append(basisLoop)
            loop = basisLoop

        # Initialise the local tensor values to 0
        initialiser = self.buildLocalTensorInitialiser()
        loop.body().prepend(initialiser)
        
        # Put the gauss loop inside the local tensor loop nest
        loop.append(gaussLoop)

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

        # One loop over the basis functions of the scalar element
        basisLoop = buildIndexForLoop(buildQuadratureBasisIndex(0, extract_subelement(coeff)))
        loop.append(basisLoop)

        # Add the expression to compute the value inside the basis loop
        basisLoop.append(self.buildQuadratureExpression(coeff))

    def buildQuadratureExpression(self, coeff):
        "Build the expression to evaluate a particular coefficient."
        rhs = self._quadratureExpressionBuilder.build(coeff)

        lhs = self._expressionBuilder.buildCoeffQuadratureAccessor(coeff, True)
        expr = PlusAssignmentOp(lhs, rhs)

        return expr

    def buildQuadratureLoopNest(self, integrand):
        "Build quadrature loop nest evaluating all coefficients of the integrand."

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

    def buildLocalTensorInitialiser(self):
        lhs = self._expressionBuilder.buildLocalTensorAccessor(self._form_data)
        rhs = Literal(0.0)
        initialiser = AssignmentOp(lhs, rhs)
        return initialiser

    def buildSubExprInitialiser(self, tree):
        lhs = self._expressionBuilder.sub_expr(tree)
        rhs = Literal(0.0)
        return InitialisationOp(lhs, rhs)

    def buildCoeffQuadratureInitialiser(self, coeff):
        accessor = self._expressionBuilder.buildCoeffQuadratureAccessor(coeff, True)
        initialiser = AssignmentOp(accessor, Literal(0.0))
        return initialiser

    def buildCoeffQuadDeclarations(self, integrand):
        
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
        extents = [Literal(self.numGaussPoints)] + [Literal(self.numDimensions)]*rank
        return Declaration(Variable(name, Array(Real(), extents)))

    def _buildBasisTensors(self):
        """When using a basis that is a tensor product of the scalar basis, we
        need to create an array that holds the tensor product. This function
        generates the code to declare and initialise that array."""

        # Build constant initialisers for shape functions/derivatives and
        # quadrature weights
        fd = self._form_data
        element = fd.coordinates.element()
        # We need to construct a fake argument and derivative to get the
        # proper names for the shape functions and derivatives
        from ufl.objects import i
        fakeArgument = Argument(element)
        fakeDerivative = SpatialDerivative(fakeArgument,i)
        nName = buildArgumentName(fakeArgument)
        dnName = buildSpatialDerivativeName(fakeDerivative)
        # Initialiser for shape functions on coordinate reference element
        nInit = buildConstArrayInitializer(nName, element._n)
        # Initialiser for shape derivatives on coordinate reference element
        dnInit = buildConstArrayInitializer(dnName, element._dn)
        # Initialiser for quadrature points on coordinate reference element
        wInit = buildConstArrayInitializer("w", element._weight)

        initialisers = [nInit, dnInit, wInit]

        for argument in uniqify(fd.arguments, lambda x: x.element()):
            # Ignore scalars
            if isinstance(argument.element(), FiniteElement):
                continue
            elif isinstance(argument.element(), VectorElement):
                n = buildVectorArgumentName(argument)
                nn = self.numNodesPerEle
                nd = self.numDimensions
                t = zeros([nd,nn*nd,self.numGaussPoints])
                # Construct initialiser lists for tensor product of scalar basis.
                for d in range(nd):
                    t[d][d*nn:(d+1)*nn][:] = element._n
                initialisers.append(buildConstArrayInitializer(n, t))
            else:
                raise RuntimeError("Tensor elements are not yet supported.")

        return initialisers

    def _buildKernelParameters(self, integrand, timestep, statutoryParameters = None):
        
        formalParameters = statutoryParameters or []
        # We always have local tensor to tabulate and timestep as parameters
        formalParameters += [self._buildLocalTensorParameter(integrand), timestep]

        # Build a parameter for each coefficient encoutered in the form
        for coeff in self._form_data.coefficients:
            param = self._buildCoefficientParameter(coeff)
            formalParameters.append(param)

        return formalParameters

    def subscript_weights(self):
        indices = [buildGaussIndex(self.numGaussPoints)]
        return indices

# Variables used globally

weights = Variable("w", Pointer(Real()) )
localTensor = Variable("localTensor", Pointer(Real()) )

# vim:sw=4:ts=4:sts=4:et
