from backends import *
from ufl.algorithms.transformations import Transformer
from ufl.algorithms.preprocess import preprocess

statutoryParameters = [ localTensor, numElements, timestep, detwei ]
threadCount = Variable("THREAD_COUNT")
threadId = Variable("THREAD_ID")

class KernelParameterComputer(Transformer):

    def compute(self, tree):
        self._parameters = list(statutoryParameters)
	self.visit(tree)
	return self._parameters

    def component_tensor(self, tree, *ops):
        pass

    def indexed(self, three, *ops):
        pass

    def index_sum(self, tree, *ops):
        pass

    def sum(self, tree, *ops):
        pass

    def product(self, tree, *ops):
        pass

    def spatial_derivative(self, tree):
        name = buildSpatialDerivativeName(tree)
	parameter = Variable(name, Pointer(Real()))
	self._parameters.append(parameter)

    def argument(self, tree):
        name = buildArgumentName(tree)
	parameter = Variable(name, Pointer(Real()))
	self._parameters.append(parameter)

def buildParameterList(tree):
    KPC = KernelParameterComputer()
    params = KPC.compute(tree)
    paramList = ParameterList(params)
    return paramList

class LoopNestBuilder(Transformer):

    def build(self, form):
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

	# Determine how many dimension loops we need by inspection
	self._indexSumDepth = 0
	self._maxIndexSumDepth = 0
	self.visit(integrand)

	# Add loops for each dimension as necessary. 
        for d in range(self._maxIndexSumDepth):
	    indVarName = dimInductionVariable(d)
	    dimLoop = buildSimpleForLoop(indVarName, numDimensions)
	    loop.append(dimLoop)
	    loop = dimLoop

	# Hand back the outer loop, so it can be inserted into some
	# scope.
	return outerLoop

    # We count the nesting depth of IndexSums to determine
    # how many dimension loops we need.
    def index_sum(self, tree):
	
	summand, indices = tree.operands()

        self._indexSumDepth = self._indexSumDepth + 1
	if self._indexSumDepth > self._maxIndexSumDepth:
	    self._maxIndexSumDepth = self._indexSumDepth

	self.visit(summand)

	self._indexSumDepth = self._indexSumDepth - 1

    # We don't care about any other node
    def expr(self, tree, *ops):
        pass

    def terminal(self, tree):
        pass

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

def buildLoopNest(form):
    LNB = LoopNestBuilder()
    return LNB.build(form)

def buildKernel(form):

    if form.form_data() is None:
        form = preprocess(form)

    integrand = form.integrals()[0].integrand()
    rank = form.form_data().rank
    t = Void()
    name = "kernel" # Fix later
    params = buildParameterList(integrand)
    #expression = buildExpression(integrand)
    loop = buildLoopNest(form)

    # Initialise the local tensor values to 0
    initialiser = buildLocalTensorInitialiser(form)
    depth = rank + 1 # Rank + element loop
    innerLoopBody = getScopeFromNest(loop, depth)
    innerLoopBody.prepend(initialiser)

    # Build the function with the loop nest inside
    statements = [loop]
    body = Scope(statements)
    kernel = FunctionDefinition(t, name, params, body)
    return kernel

def getScopeFromNest(nest, depth):
    body = nest.body()
    # Descend through the bodies until we reach the correct one
    for i in range(1,depth):
        loop = body.find(lambda x: isinstance(x, ForLoop))
	body = loop.body()
    return body

def buildLocalTensorInitialiser(form):
    form_data = form.form_data()
    rank = form_data.rank
    # First index is the element index
    indices = [ElementIndex()]

    # One rank index for each rank
    for r in range(rank):
        indices.append(RankIndex(r))
    offset = buildOffset(indices)
    
    # Initialise this element to 0.0
    lhs = Subscript(localTensor, offset)
    rhs = Literal(0.0)
    initialiser = AssignmentOp(lhs, rhs)
    return initialiser

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

# Global variables for code generation.
# Eventually these need to be set by the caller of the code generator
numNodesPerEle = 3
numDimensions = 2
numGaussPoints = 6
