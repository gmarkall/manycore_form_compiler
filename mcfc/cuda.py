from backends import *
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
        name = buildCoefficientQuadName(tree)
	base = Variable(name)
	
	# Build the subscript.
	indices = [GaussIndex()]
	offset = buildOffset(indices)
	coeffExpr = Subscript(base, offset)
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
    count = tree.count()
    name = '%s_%d' % (element.shortstr(), count)
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
	return self._parameters

    # The expression structure does not affect the parameters.
    def expr(self, tree, *ops):
        pass

#    def component_tensor(self, tree, *ops):
#        pass
#
#    def indexed(self, three, *ops):
#        pass
#
#    def index_sum(self, tree, *ops):
#        pass
#
#    def sum(self, tree, *ops):
#        pass
#
#    def product(self, tree, *ops):
#        pass
#
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
    indVarName = rankInductionVariable(0)
    rankLoop = buildSimpleForLoop(indVarName, numNodesPerEle)
    loop.append(rankLoop)
    loop = rankLoop

    # Add another loop for each rank of the form (probably no
    # more than one more... )
    for r in range(1,rank):
	indVarName = rankInductionVariable(r)
	rankLoop = buildSimpleForLoop(indVarName, numNodesPerEle)
	loop.append(rankLoop)
	loop = rankLoop
    
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

def buildElementLoop():
    indVarName = eleInductionVariable()
    var = Variable(indVarName, Integer())
    init = AssignmentOp(var, threadId)
    test = LessThanOp(var, numElements)
    inc = PlusAssignmentOp(var, threadCount)
    ast = ForLoop(init, test, inc)
    return ast

def eleInductionVariable():
    return "i_ele"

def gaussInductionVariable():
    return "i_g"

def rankInductionVariable(count):
    name = "i_r_%d" % (count)
    return name

def dimInductionVariable(count):
    name = "i_d_%d" % (count)
    return name

def buildKernel(form):

    if form.form_data() is None:
        form = preprocess(form)

    integrand = form.integrals()[0].integrand()
    rank = form.form_data().rank
    
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

class CodeIndex:

    def __init__(self, count=None):
        self._count = count

class RankIndex(CodeIndex):
    
    def extent(self):
        return Literal(numNodesPerEle)

    def name(self):
        return rankInductionVariable(self._count)

class ElementIndex(CodeIndex):

    def extent(self):
        return numElements

    def name(self):
        return eleInductionVariable()

class GaussIndex(CodeIndex):

    def extent(self):
        return Literal(numGaussPoints)

    def name(self):
        return gaussInductionVariable()

class DimIndex(CodeIndex):

    def extent(self):
        return Literal(numDimensions)

    def name(self):
        return dimInductionVariable(self._count)

# Global variables for code generation.
# Eventually these need to be set by the caller of the code generator
numNodesPerEle = 3
numDimensions = 2
numGaussPoints = 6
