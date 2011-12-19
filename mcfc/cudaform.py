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


# MCFC libs
from form import *
from cudaparameters import CudaKernelParameterGenerator, numElements, statutoryParameters
from cudaexpression import CudaExpressionBuilder, CudaQuadratureExpressionBuilder, buildElementLoop, ElementIndex
# UFL libs
from ufl.finiteelement import FiniteElement, VectorElement, TensorElement

class CudaFormBackend(FormBackend):

    def __init__(self):
        FormBackend.__init__(self)
        self._expressionBuilder = CudaExpressionBuilder(self)
        self._quadratureExpressionBuilder = CudaQuadratureExpressionBuilder(self)

    def compile(self, name, form):
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

        basisTensors = self._buildBasisTensors(form_data)

        # Build the loop nest
        loopNest = self.buildLoopNest(form)

        # Initialise the local tensor values to 0
        initialiser = self.buildLocalTensorInitialiser(form)
        depth = rank + 1 # Rank + element loop
        loopBody = getScopeFromNest(loopNest, depth)
        loopBody.prepend(initialiser)

        # Insert the expressions into the loop nest
        partitions = findPartitions(integrand)
        for (tree, depth) in partitions:
            expression = self.buildExpression(form, tree)
            exprDepth = depth + rank + 2 # 2 = Ele loop + gauss loop
            loopBody = getScopeFromNest(loopNest, exprDepth)
            loopBody.prepend(expression)

        # Build the function with the loop nest inside
        statements = basisTensors + [loopNest]
        body = Scope(statements)
        kernel = FunctionDefinition(Void(), name, formalParameters, body)
        
        # If there's any coefficients, we need to build a loop nest
        # that calculates their values at the quadrature points
        if form_data.num_coefficients > 0:
            declarations = self.buildCoeffQuadDeclarations(form)
            quadLoopNest = self.buildQuadratureLoopNest(form)
            loopNest.prepend(quadLoopNest)
            for decl in declarations:
                loopNest.prepend(decl)
        
        # Make this a Cuda kernel.
        kernel.setCudaKernel(True)
        return kernel

    def _buildCoeffQuadDeclaration(self, name, rank):
        length = self.numGaussPoints * pow(self.numDimensions, rank)
        return Declaration(Variable(name, Array(Real(), Literal(length))))

    def _buildKernelParameters(self, tree, form):
        KPG = CudaKernelParameterGenerator()
        return KPG.generate(tree, form, statutoryParameters)

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

    def buildLoopNest(self, form):
        "Build the loop nest for evaluating a form expression."
        form_data = form.form_data()
        rank = form_data.rank

        # FIXME what if we have multiple integrals?
        integrand = form.integrals()[0].integrand()

        # The element loop is the outermost loop
        loop = buildElementLoop()
        outerLoop = loop

        # Build the loop over the first rank, which always exists
        indVarName = self.buildBasisIndex(0).name()
        basisLoop = buildSimpleForLoop(indVarName, self._numBasisFunctions(form))
        loop.append(basisLoop)
        loop = basisLoop

        # Add another loop for each rank of the form (probably no
        # more than one more... )
        for r in range(1,rank):
            indVarName = self.buildBasisIndex(r).name()
            basisLoop = buildSimpleForLoop(indVarName, self._numBasisFunctions(form))
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

    def subscript_detwei(self):
        indices = [ElementIndex(), self.buildGaussIndex()]
        return indices

# vim:sw=4:ts=4:sts=4:et
