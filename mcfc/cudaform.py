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
        indices = [ElementIndex(), RankIndex(count), GaussIndex()]
	return indices

    def subscript_SpatialDerivative(self,tree,depth):
	# Build the subscript based on the argument count and the
	# nesting depth of IndexSums of the expression.
	argument = tree.operands()[0]
	count = argument.count()
	indices = [ElementIndex(), RankIndex(count), GaussIndex(), DimIndex(depth)]
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
	    indices.append(RankIndex(r))

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
	indices = [RankIndex(0)]
	for r in range(rank):
	    index = DimIndex(r)
	    indices.insert(0, index)
	return indices

# For generating the kernel parameter list

class KernelParameterComputer(Transformer):

    def compute(self, tree):
        self._parameters = list(statutoryParameters)
	self.visit(tree)
	self._parameters = uniqify(self._parameters)
	return self._parameters

    # The expression structure does not affect the parameters.
    def expr(self, tree, *ops):
        pass

    def spatial_derivative(self, tree):
        name = buildSpatialDerivativeName(tree)
	parameter = Variable(name, Pointer(Real()))
	self._parameters.append(parameter)

    def argument(self, tree):
        name = buildArgumentName(tree)
	parameter = Variable(name, Pointer(Real()))
	self._parameters.append(parameter)

    def coefficient(self, tree):
        name = buildCoefficientName(tree)
	parameter = Variable(name, Pointer(Real()))
	self._parameters.append(parameter)

def buildCoeffQuadDeclarations(form):
    form_data = form.form_data()
    coefficients = form_data.coefficients
    declarations = []

    for coeff in coefficients:
        name = buildCoefficientQuadName(coeff)
	rank = coeff.rank()
	length = numGaussPoints * pow(numDimensions, rank)
	t = Array(Real(), Literal(length))
	t.setCudaShared(True)
	var = Variable(name, t)
	decl = Declaration(var)
	declarations.append(decl)

    return declarations

class CudaFormBackend(FormBackend):

    def __init__(self):
        self._expressionBuilder = CudaExpressionBuilder()
	self._quadratureExpressionBuilder = CudaQuadratureExpressionBuilder()
	self._kernelParameterComputer = KernelParameterComputer()

    def compile(self, form):

	if form.form_data() is None:
	    form = preprocess(form)

	integrand = form.integrals()[0].integrand()
	form_data = form.form_data()
	rank = form_data.rank
	
	# Things for kernel declaration.
	t = Void()
	name = "kernel" # Fix later
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
	    declarations = buildCoeffQuadDeclarations(form)
	    quadLoopNest = self.buildQuadratureLoopNest(form)
	    loopNest.prepend(quadLoopNest)
	    for decl in declarations:
		loopNest.prepend(decl)
	
	# Make this a Cuda kernel.
	kernel.setCudaKernel(True)
	return kernel

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
	ISC = IndexSumCounter()
	numDimLoops = ISC.count(integrand)

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
	params = self._kernelParameterComputer.compute(tree)
	paramList = ParameterList(params)
	return paramList


