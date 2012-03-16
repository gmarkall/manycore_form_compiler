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
from utilities import uniqify

class FormBackend(object):
    "Base class for generating form tabulation kernels."

    def __init__(self):
        self._expressionBuilder = None
        self._quadratureExpressionBuilder = None

    def compile(self, form, outerScope = None):
        "Compile a pre-processed form."

        # FIXME what if we have multiple integrals?
        integrand = form.integrals()[0].integrand()
        form_data = form.form_data()
        assert form_data, "Form has no form data attached!"
        rank = form_data.rank

        # Element data queried by femtools
        self.elementdata = {}
        # Static array initialisers for shape functions/derivatives of elements
        self.initialisers = {}

        # Initialise basis tensors if necessary
        self._buildBasisTensors(form)

        # Build the loop nest
        loopNest, gaussBody = self.buildExpressionLoopNest(form)

        # Insert the expressions into the loop nest
        partitions = findPartitions(integrand)
        loopBody = gaussBody
        for (tree, indices) in partitions:
            expression, listexpressions = self.buildExpression(form, tree)
            buildLoopNest(loopBody, indices).prepend(expression)
            for expr in listexpressions:
                loopBody.prepend(expr)

        # Declare temporary variables to hold subexpression values
        for (tree, _) in partitions:
            initialiser = self.buildSubExprInitialiser(tree)
            loopBody.prepend(initialiser)
        
        # Insert the local tensor expression into the loop nest. There should
        # be no listexpressions.
        expression, _ = self.buildLocalTensorExpression(form, integrand)
        loopBody.append(expression)

        # If there's any coefficients, we need to build a loop nest
        # that calculates their values at the quadrature points
        # Note: this uses data generated during building of the expressions,
        # hence needs to be done afterwards, though it comes first in the
        # generated code
        coeff_decls, coeff_loopnest = [], []
        if form_data.num_coefficients > 0:
            coeff_decls, coeff_loopnest = self.buildQuadrature(form)

        k = lambda x:x._lhs.unparse()
        declarations = sorted(self.initialisers.values(), key=k) + coeff_decls
        statements = coeff_loopnest + [loopNest]

        # If we are given an outer scope, append the statements to it
        if outerScope:
            for s in statements:
                outerScope.append(s)
            statements = [outerScope]

        # Get parameter list for kernel declaration.
        formalParameters = self._buildKernelParameters(form)

        # Build the function with the loop nest inside
        body = Scope(declarations + statements)
        kernel = FunctionDefinition(Void(), form_data.name, formalParameters, body)

        return kernel

    def buildLocalTensorExpression(self, form, tree):
        return self.buildExpression(form, tree, True)

    def buildExpression(self, form, tree, localTensor=False):
        "Build the expression represented by the subtree tree of form."
        # If the tree is rooted by a SubExpr, we need to construct the 
        # expression beneath it.
        expr = tree if localTensor else tree.operands()[0]
        rhs, listexpr = self._expressionBuilder.build(expr)

        if localTensor:
            # The rhs of a form needs to be multiplied by the quadrature weights
            indices = self.subscript_weights()
            weightsExpr = self._expressionBuilder.buildSubscript(weights, indices)
            rhs = MultiplyOp(rhs, weightsExpr)
            # Assign expression to the local tensor
            lhs = self._expressionBuilder.buildLocalTensorAccessor(form)
        else:
            # Assign expression to the correct Subexpression variable
            lhs = self._expressionBuilder.sub_expr(tree)
        
        expr = PlusAssignmentOp(lhs, rhs)
        return expr, listexpr

    def buildExpressionLoopNest(self, form):
        "Build the loop nest for evaluating a form expression."

        # FIXME what if we have multiple integrals?
        integrand = form.integrals()[0].integrand()

        # Build a loop for the quadrature
        gaussLoop = buildIndexForLoop(buildGaussIndex(self.numGaussPoints))
        
        # Loops over the indices of the local tensor
        outerLoop = self.buildLocalTensorLoops(form, gaussLoop)
        
        # Hand back the outer loop, so it can be inserted into some
        # scope.
        return outerLoop, gaussLoop.body()

    def buildLocalTensorLoops(self, form, gaussLoop):
        # Build the loop over the first rank, which always exists
        loop = buildIndexForLoop(buildBasisIndex(0, form))
        outerLoop = loop

        # Add a loop over basis functions for each rank of the form
        for r in range(1,form.form_data().rank):
            basisLoop = buildIndexForLoop(buildBasisIndex(r, form))
            loop.append(basisLoop)
            loop = basisLoop

        # Initialise the local tensor values to 0
        initialiser = self.buildLocalTensorInitialiser(form)
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

    def buildQuadrature(self, form):
        """Build quadrature loop nest evaluating all coefficients of the form
        and their declarations."""

        # FIXME what if we have multiple integrals?
        integrand = form.integrals()[0].integrand()
        coefficients, spatialDerivatives = CoefficientUseFinder().find(integrand)

        declarations = []

        # Outer loop over gauss points
        gaussLoop = buildIndexForLoop(buildGaussIndex(self.numGaussPoints))

        # Build a loop nest for each coefficient containing expressions
        # to compute its value
        for coeff in coefficients:
            rank = coeff.rank()
            # declaration
            name = buildCoefficientQuadName(coeff)
            decl = self._buildCoeffQuadDeclaration(name, rank)
            declarations.append(decl)
            # loop nest
            self.buildCoefficientLoopNest(coeff, rank, gaussLoop)

        for spatialDerivative in spatialDerivatives:
            operand = spatialDerivative.operands()[0]
            rank = operand.rank() + 1
            # delaration
            name = buildSpatialDerivativeName(spatialDerivative)
            decl = self._buildCoeffQuadDeclaration(name, rank)
            declarations.append(decl)
            # loop nest
            self.buildCoefficientLoopNest(spatialDerivative, rank, gaussLoop)

        return declarations, [gaussLoop]

    def buildLocalTensorInitialiser(self, form):
        lhs = self._expressionBuilder.buildLocalTensorAccessor(form)
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

    def _buildCoeffQuadDeclaration(self, name, rank):
        extents = [Literal(self.numGaussPoints)] + [Literal(self.numDimensions)]*rank
        return Declaration(Variable(name, Array(Real(), extents)))

    def addInitialiser(self, element, buildarray, buildname = lambda n: n):

        def buildTensorProduct(e, buildarray):
            t = zeros([e.numDimensions, e.numNodesPerEle*e.numDimensions, e.numGaussPoints])
            # Construct initialiser lists for tensor product of scalar basis.
            for d in range(e.numDimensions):
                t[d][d*e.numNodesPerEle:(d+1)*e.numNodesPerEle][:] = buildarray(e)
            return t

        if isinstance(element, FiniteElement):
            name = buildname(elementName(element))
            getarray = buildarray
        elif isinstance(element, VectorElement):
            name = buildname(vectorName(elementName(element)))
            getarray = lambda e: buildTensorProduct(e, buildarray)
        else:
            raise NotImplementedError("Tensor elements are not yet supported.")

        if name not in self.initialisers:
            # Only query femtools for elements if we haven't already done so
            if element not in self.elementdata:
                self.elementdata[element] = FemtoolsElement(element)
            self.initialisers[name] = buildConstArrayInitializer(name,
                                        getarray(self.elementdata[element]))

    def _buildBasisTensors(self, form):
        """When using a basis that is a tensor product of the scalar basis, we
        need to create an array that holds the tensor product. This function
        generates the code to declare and initialise that array."""

        form_data = form.form_data()
        # FIXME what if we have multiple integrals?
        integrand = form.integrals()[0].integrand()
        arguments, spatialDerivatives = ArgumentUseFinder().find(integrand)

        # Build constant initialisers for shape derivatives and quadrature
        # weights on coordinate field (we need to use the scalar element)
        # FIXME: We only look at the element of the coordinate field for now
        coord_element = form_data.coordinates.element().sub_elements()[0]

        # Initialiser for quadrature points on coordinate reference element
        self.addInitialiser(coord_element, lambda e: e.weights, lambda n: 'w')
        # Initialiser for shape derivatives on coordinate reference element
        self.addInitialiser(coord_element, lambda e: e.dn, derivativeName)

        # Build shape function initialisers for arguments used in the form
        for argument in arguments:
            self.addInitialiser(argument.element(), lambda e: e.n)

        # Build shape derivative initialisers for argument derivatives used in the form
        for deriv in spatialDerivatives:
            self.addInitialiser(deriv.operands()[0].element(), lambda e: e.dn, derivativeName)

        # Build shape function initialisers for elements of coefficients (other
        # than the Jacobian) used in the form. These need the shape functions
        # for interpolation of the coefficient field at quadrature points.
        for coeff in form_data.coefficients:
            if not isJacobian(coeff):
                self.addInitialiser(extract_subelement(coeff), lambda e: e.n)

        # Set basic properties of the element
        # FIXME: We only look at the element of the coordinate field for now
        # FIXME: This should be set someplace else
        coord_felement = self.elementdata[coord_element]
        self.numNodesPerEle = coord_felement.numNodesPerEle
        self.numDimensions = coord_felement.numDimensions
        self.numGaussPoints = coord_felement.numGaussPoints

        return [buildConstArrayInitializer(name, arr) for name, arr in \
                sorted(self.initialisers.items(), key=lambda x:x[0])]

    def _buildKernelParameters(self, form, statutoryParameters = None):
        
        formalParameters = statutoryParameters or []
        # We always have local tensor to tabulate and timestep as parameters
        formalParameters += [self._buildLocalTensorParameter(form), timestep]

        # Build a parameter for each coefficient encoutered in the form
        for coeff in form.form_data().coefficients:
            param = self._buildCoefficientParameter(coeff)
            formalParameters.append(param)

        return formalParameters

    def subscript_weights(self):
        indices = [buildGaussIndex(self.numGaussPoints)]
        return indices

# Variables used globally

weights = Variable("w", Pointer(Real()) )
timestep = Variable("dt", Real() )
localTensor = Variable("localTensor", Pointer(Real()) )

# vim:sw=4:ts=4:sts=4:et
