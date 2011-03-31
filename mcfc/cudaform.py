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
from utilities import uniqify
# FEniCS UFL libs
from ufl.algorithms.transformations import Transformer
from ufl.algorithms.preprocess import preprocess

# Variables

numElements = Variable("n_ele", Integer() )

statutoryParameters = [ localTensor, numElements, timestep, detwei ]

threadCount = Variable("THREAD_COUNT")
threadId = Variable("THREAD_ID")

# The ElementIndex is here and not form.py because not all backends need
# an element index (e.g. OP2).

class ElementIndex(CodeIndex):

    def extent(self):
        return numElements

    def name(self):
        return eleInductionVariable()

def eleInductionVariable():
    return "i_ele"

# Expression builders

class CudaExpressionBuilder(ExpressionBuilder):

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
        argument = tree.operands()[0]
        count = argument.count()
        indices = [ElementIndex(), DimIndex(depth), GaussIndex(), BasisIndex(count)]
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
        for r in range(depth): # Need to add one, since depth started at -1
            indices.append(DimIndex(r))
        
        return indices

class CudaQuadratureExpressionBuilder(QuadratureExpressionBuilder):

    def subscript(self, tree):
        rank = tree.rank()
        indices = [ElementIndex(), BasisIndex(0)]
        for r in range(rank):
            index = DimIndex(r)
            indices.insert(0, index)
        return indices

    def subscript_argument(self, tree):
        # The count of the basis function induction variable is always
        # 0 in the quadrature loops (i.e. i_r_0)
        indices = [BasisIndex(0), GaussIndex()]
        return indices

# For generating the kernel parameter list

class KernelParameterComputer(Transformer):

    def compute(self, tree):
        parameters = list(statutoryParameters)
        
        self._coefficients = []
        self._arguments = []
        self._spatialDerivatives = []
        
        self.visit(tree)
        
        self._coefficients = uniqify(self._coefficients)
        self._arguments = uniqify(self._arguments)
        self._spatialDerivatives = uniqify(self._spatialDerivatives)
        
        parameters.extend(self._coefficients)
        parameters.extend(self._arguments)
        parameters.extend(self._spatialDerivatives)
        return parameters

    # The expression structure does not affect the parameters.
    def expr(self, tree, *ops):
        pass

    def spatial_derivative(self, tree):
        name = buildSpatialDerivativeName(tree)
        parameter = Variable(name, Pointer(Real()))
        self._spatialDerivatives.append(parameter)

    def argument(self, tree):
        name = buildArgumentName(tree)
        parameter = Variable(name, Pointer(Real()))
        self._arguments.append(parameter)

    def coefficient(self, tree):
        name = buildCoefficientName(tree)
        parameter = Variable(name, Pointer(Real()))
        self._coefficients.append(parameter)

class CudaFormBackend(FormBackend):

    def __init__(self):
        self._expressionBuilder = CudaExpressionBuilder()
        self._quadratureExpressionBuilder = CudaQuadratureExpressionBuilder()
        self._kernelParameterComputer = KernelParameterComputer()
        self._indexSumCounter = IndexSumCounter()

    def compile(self, name, form):

        if form.form_data() is None:
            form = preprocess(form)

        integrand = form.integrals()[0].integrand()
        form_data = form.form_data()
        rank = form_data.rank
        
        # Things for kernel declaration.
        t = Void()
        params = self.buildParameterList(integrand)
        
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
        
        # Make this a Cuda kernel.
        kernel.setCudaKernel(True)
        return kernel

    def buildCoeffQuadDeclarations(self, form):
        # The FormBackend's list of variables to declare is
        # fine, but we want them to be __shared__
        declarations = FormBackend.buildCoeffQuadDeclarations(self, form)
        for decl in declarations:
            decl.setCudaShared(True)
        return declarations

    def buildQuadratureLoopNest(self, form):
        
        form_data = form.form_data()
        coefficients = form_data.coefficients

        # Outer loop over gauss points
        indVar = gaussInductionVariable()
        gaussLoop = buildSimpleForLoop(indVar, numGaussPoints)

        # Build a loop nest for each coefficient containing expressions
        # to compute its value
        for coeff in coefficients:
            rank = coeff.rank()
            loop = gaussLoop

            # Build loop over the correct number of dimensions
            for r in range(rank):
                indVar = dimInductionVariable(r)
                dimLoop = buildSimpleForLoop(indVar, numDimensions)
                loop.append(dimLoop)
                loop = dimLoop

            # Add initialiser here
            initialiser = self.buildCoeffQuadratureInitialiser(coeff)
            loop.append(initialiser)

            # One loop over the basis functions
            indVar = basisInductionVariable(0)
            basisLoop = buildSimpleForLoop(indVar, numNodesPerEle)
            loop.append(basisLoop)
        
            # Add the expression to compute the value inside the basis loop
            computation = self.buildQuadratureExpression(coeff)
            basisLoop.append(computation)

            depth = rank + 1 # Plus the loop over basis functions

        return gaussLoop

    def buildLoopNest(self, form):
        form_data = form.form_data()
        rank = form_data.rank
        integrand = form.integrals()[0].integrand()

        # The element loop is the outermost loop
        loop = self.buildElementLoop()
        outerLoop = loop

        # Build the loop over the first rank, which always exists
        indVarName = basisInductionVariable(0)
        basisLoop = buildSimpleForLoop(indVarName, numNodesPerEle)
        loop.append(basisLoop)
        loop = basisLoop

        # Add another loop for each rank of the form (probably no
        # more than one more... )
        for r in range(1,rank):
            indVarName = basisInductionVariable(r)
            basisLoop = buildSimpleForLoop(indVarName, numNodesPerEle)
            loop.append(basisLoop)
            loop = basisLoop
        
        # Add a loop for the quadrature
        indVarName = gaussInductionVariable()
        gaussLoop = buildSimpleForLoop(indVarName, numGaussPoints)
        loop.append(gaussLoop)
        loop = gaussLoop

        # Determine how many dimension loops we need by inspection.
        # We count the nesting depth of IndexSums to determine
        # how many dimension loops we need.
        numDimLoops = self._indexSumCounter.count(integrand)

        # Add loops for each dimension as necessary. 
        for d in range(numDimLoops):
            indVarName = dimInductionVariable(d)
            dimLoop = buildSimpleForLoop(indVarName, numDimensions)
            loop.append(dimLoop)
            loop = dimLoop

        # Hand back the outer loop, so it can be inserted into some
        # scope.
        return outerLoop

    def buildElementLoop(self):
        indVarName = eleInductionVariable()
        var = Variable(indVarName, Integer())
        init = InitialisationOp(var, threadId)
        test = LessThanOp(var, numElements)
        inc = PlusAssignmentOp(var, threadCount)
        ast = ForLoop(init, test, inc)
        return ast

    def buildParameterList(self,tree):
        return self._kernelParameterComputer.compute(tree)

# vim:sw=4:ts=4:sts=4:et
