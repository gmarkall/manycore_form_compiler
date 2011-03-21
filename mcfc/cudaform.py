# Python libs
import sys
# MCFC libs
from form import *
# FEniCS UFL libs
from ufl.algorithms.transformations import Transformer
from ufl.algorithms.preprocess import preprocess

# Variables

numElements = Variable("n_ele", Integer() )
detwei = Variable("detwei", Pointer(Real()) )
timestep = Variable("dt", Real() )
localTensor = Variable("localTensor", Pointer(Real()) )

statutoryParameters = [ localTensor, numElements, timestep, detwei ]

threadCount = Variable("THREAD_COUNT")
threadId = Variable("THREAD_ID")

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
	indices = [ElementIndex(), GaussIndex()]
	offset = buildOffset(indices)
	detweiExpr = Subscript(detwei, offset)
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

	# Build the subscript based on the argument count and the
	# nesting depth of IndexSums of the expression.
	argument = tree.operands()[0]
	count = argument.count()
	depth = self._indexSumDepth
	indices = [ElementIndex(), RankIndex(count), GaussIndex(), DimIndex(depth)]
	offset = buildOffset(indices)
	spatialDerivExpr = Subscript(base, offset)
	self._exprStack.append(spatialDerivExpr)
 
    def argument(self, tree):
        name = buildArgumentName(tree)
        base = Variable(name)

	# Build the subscript based on the argument count
	count = tree.count()
	indices = [ElementIndex(), RankIndex(count), GaussIndex()]
	offset = buildOffset(indices)
	argExpr = Subscript(base, offset)
        self._exprStack.append(argExpr)

    def coefficient(self, tree):
	coeffExpr = buildCoeffQuadratureAccessor(tree)
	self._exprStack.append(coeffExpr)

def buildExpression(form, tree):
    "Build the expression represented by the subtree tree of form."
    # Build the rhs expression
    EB = ExpressionBuilder()
    rhs = EB.build(tree)

    # Assign expression to the local tensor value
    lhs = buildLocalTensorAccessor(form)
    expr = PlusAssignmentOp(lhs, rhs)

    return expr

def buildArgumentName(tree):
    element = tree.element()
    name = element.shortstr()
    return name

def buildSpatialDerivativeName(tree):
    argument = tree.operands()[0]
    argName = buildArgumentName(argument)
    spatialDerivName = 'd_%s' % (argName)
    return spatialDerivName

def buildCoefficientName(tree):
    count = tree.count()
    name = 'c%d' % (count)
    return name

def buildCoefficientQuadName(tree):
    count = tree.count()
    name = 'c_q%d' %(count)
    return name

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

def buildParameterList(tree):
    KPC = KernelParameterComputer()
    params = KPC.compute(tree)
    paramList = ParameterList(params)
    return paramList

def buildLoopNest(form):
    form_data = form.form_data()
    rank = form_data.rank
    integrand = form.integrals()[0].integrand()

    # The element loop is the outermost loop
    loop = buildElementLoop()
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

def buildQuadratureLoopNest(form):
    
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
        accessor = buildCoeffQuadratureAccessor(coeff)
	initialiser = AssignmentOp(accessor, Literal(0.0))
	loop.append(initialiser)

        # One loop over the basis functions
        indVar = basisInductionVariable(0)
        basisLoop = buildSimpleForLoop(indVar, numNodesPerEle)
        loop.append(basisLoop)
    
        # Add the expression to compute the value inside the basis loop
	indices = [RankIndex(0)]
	for r in range(rank):
	    index = DimIndex(r)
	    indices.insert(0, index)
        offset = buildOffset(indices)
	coeffAtBasis = Variable(buildCoefficientName(coeff))
	rhs = Subscript(coeffAtBasis, offset)
	computation = PlusAssignmentOp(accessor, rhs)
	basisLoop.append(computation)

        depth = rank + 1 # Plus the loop over basis functions

    return gaussLoop

def buildCoeffQuadratureAccessor(coeff):
    name = buildCoefficientQuadName(coeff)
    base = Variable(name)
    
    # Build the subscript based on the rank
    indices = [GaussIndex()]
    depth = coeff.rank()
    for r in range(depth): # Need to add one, since depth started at -1
	indices.append(DimIndex(r))
    offset = buildOffset(indices)

    coeffExpr = Subscript(base, offset)
    return coeffExpr

def buildElementLoop():
    indVarName = eleInductionVariable()
    var = Variable(indVarName, Integer())
    init = InitialisationOp(var, threadId)
    test = LessThanOp(var, numElements)
    inc = PlusAssignmentOp(var, threadCount)
    ast = ForLoop(init, test, inc)
    return ast

def eleInductionVariable():
    return "i_ele"

def gaussInductionVariable():
    return "i_g"

def basisInductionVariable(count):
    name = "i_r_%d" % (count)
    return name

def dimInductionVariable(count):
    name = "i_d_%d" % (count)
    return name

def buildKernel(form):

    if form.form_data() is None:
        form = preprocess(form)

    integrand = form.integrals()[0].integrand()
    form_data = form.form_data()
    rank = form_data.rank
    
    # Things for kernel declaration.
    t = Void()
    name = "kernel" # Fix later
    params = buildParameterList(integrand)
    
    # Build the loop nest
    loopNest = buildLoopNest(form)

    # Initialise the local tensor values to 0
    initialiser = buildLocalTensorInitialiser(form)
    depth = rank + 1 # Rank + element loop
    loopBody = getScopeFromNest(loopNest, depth)
    loopBody.prepend(initialiser)

    # Insert the expressions into the loop nest
    partitions = findPartitions(integrand)
    for (tree, depth) in partitions:
        expression = buildExpression(form, tree)
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
	quadLoopNest = buildQuadratureLoopNest(form)
	loopNest.prepend(quadLoopNest)
	for decl in declarations:
	    loopNest.prepend(decl)
    
    # Make this a Cuda kernel.
    kernel.setCudaKernel(True)
    return kernel

def getScopeFromNest(nest, depth):
    body = nest.body()
    # Descend through the bodies until we reach the correct one
    for i in range(1,depth):
        loop = body.find(lambda x: isinstance(x, ForLoop))
	body = loop.body()
    return body

def buildLocalTensorInitialiser(form):
    lhs = buildLocalTensorAccessor(form)
    rhs = Literal(0.0)
    initialiser = AssignmentOp(lhs, rhs)
    return initialiser

def buildLocalTensorAccessor(form):
    form_data = form.form_data()
    rank = form_data.rank
    
    # First index is the element index
    indices = [ElementIndex()]

    # One rank index for each rank
    for r in range(rank):
        indices.append(RankIndex(r))
    offset = buildOffset(indices)
    
    # Subscript the local tensor variable
    expr = Subscript(localTensor, offset)
    return expr
    

def buildOffset(indices):
    # Start our expression with the first index
    name = indices[0].name()
    offset = Variable(name)
    
    # Compute the expression for all indices
    for v in range(1,len(indices)):
        subindices = indices[:v]
	name = indices[v].name()
	expr = Variable(name)
	
	# Find the correct offset for this index
	for u in range(len(subindices)):
	    multiplier = subindices[u].extent()
	    expr = MultiplyOp(multiplier, expr)
	offset = AddOp(offset, expr)
    
    return offset

# From http://www.peterbe.com/plog/uniqifiers-benchmark. This is version f5.
def uniqify(seq, idfun=None):
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result

# Register this implementation with form.py
impl = sys.modules[__name__]
registerImplementation(impl)
