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
from op2parameters import Op2KernelParameterGenerator, _buildArrayParameter
from op2expression import Op2ExpressionBuilder, Op2QuadratureExpressionBuilder

class Op2FormBackend(FormBackend):

    def __init__(self):
        FormBackend.__init__(self)
        self._expressionBuilder = Op2ExpressionBuilder(self)
        self._quadratureExpressionBuilder = Op2QuadratureExpressionBuilder(self)

    def compile(self, name, form):

        # FIXME what if we have multiple integrals?
        integrand = form.integrals()[0].integrand()
        form_data = form.form_data()
        assert form_data, "Form has no form data attached!"
        rank = form_data.rank
        
        # Things for kernel declaration.
        t = Void()
        params = self.buildParameterList(integrand, form)
        
        # Build the loop nest
        loopNest = self.buildLoopNest(form)

        # Initialise the local tensor values to 0
        initialiser = self.buildLocalTensorInitialiser(form)
        depth = rank
        loopBody = getScopeFromNest(loopNest, depth)
        loopBody.prepend(initialiser)

        # Insert the expressions into the loop nest
        partitions = findPartitions(integrand)
        for (tree, depth) in partitions:
            expression = self.buildExpression(form, tree)
            exprDepth = depth + rank + 1 # 1 = gauss loop
            loopBody = getScopeFromNest(loopNest, exprDepth)
            loopBody.prepend(expression)

        # Build the function with the loop nest inside
        statements = [loopNest]
        body = Scope(statements)
        kernel = FunctionDefinition(t, name, params, body)
        
        # If there's any coefficients, we need to build a loop nest
        # that calculates their values at the quadrature points
        if form_data.num_coefficients > 0:
            declarations = self.buildCoeffQuadDeclarations(form)
            quadLoopNest = self.buildQuadratureLoopNest(form)
            loopNest.prepend(quadLoopNest)
            for decl in declarations:
                loopNest.prepend(decl)
        
        return kernel

    def _buildCoeffQuadDeclaration(self, name, rank):
        extents = [Literal(self.numGaussPoints)] + [Literal(self.numDimensions)]*rank
        return Declaration(Variable(name, Array(Real(), extents)))

    def _buildKernelParameters(self, tree, form):
        KPG = Op2KernelParameterGenerator(self)

        detwei = _buildArrayParameter("detwei", KPG.expBuilder.subscript_detwei())
        timestep = Variable("dt", Real() )
        localTensor = _buildArrayParameter("localTensor", KPG.expBuilder.subscript_LocalTensor(form))

        statutoryParameters = [ localTensor, timestep, detwei ]

        return KPG.generate(tree, form, statutoryParameters)

    def buildCoefficientLoopNest(self, coeff, rank, scope):

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
        form_data = form.form_data()
        rank = form_data.rank
        # FIXME what if we have multiple integrals?
        integrand = form.integrals()[0].integrand()

        # Build the loop over the first rank, which always exists
        indVarName = self.buildBasisIndex(0).name()
        loop = buildSimpleForLoop(indVarName, self.numNodesPerEle)
        outerLoop = loop

        # Add another loop for each rank of the form (probably no
        # more than one more... )
        for r in range(1,rank):
            indVarName = self.buildBasisIndex(r).name()
            basisLoop = buildSimpleForLoop(indVarName, self.numNodesPerEle)
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

    def buildParameterList(self, tree, form):
        formalParameters, _ = self._buildKernelParameters(tree, form)
        return formalParameters

# vim:sw=4:ts=4:sts=4:et
