from backends import *
from ufl.algorithms.transformations import Transformer
from ufl.algorithms.preprocess import preprocess

statutoryParameters = [ localTensor, numElements, timestep, detwei ]

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

	# Build the loop over the first rank, which always exists
	indVarName = rankInductionVariable(0)
	outerLoop = buildSimpleForLoop(indVarName, numNodesPerEle)
	loop = outerLoop

	# Add another loop for each rank of the form (probably no
	# more than one more... )
	for r in range(1,rank):
	    indVarName = rankInductionVariable(r)
	    newLoop = buildSimpleForLoop(indVarName, numNodesPerEle)
	    loop.append(newLoop)
	    loop = newLoop
	
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
	    newLoop = buildSimpleForLoop(indVarName, numDimensions)
	    loop.append(newLoop)
	    loop = newLoop

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
    expression = buildExpression(integrand)
    loop = buildLoopNest(form)
    innerLoopBody = getScopeFromNest(loop, rank)
    innerLoopBody.append(expression)
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

# Global variables for code generation.
# Eventually these need to be set by the caller of the code generator
numNodesPerEle = 3
numDimensions = 2
numGaussPoints = 6
