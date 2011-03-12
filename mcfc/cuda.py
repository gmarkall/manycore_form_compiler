from backends import *
from ufl.algorithms.transformations import Transformer

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

    def build(self, tree):
        form_data = tree.form_data()
	rank = form_data.rank
	indVarName = 'i_r_0'
	outerLoop = buildSimpleForLoop(indVarName, numNodesPerEle)
	loop = outerLoop
	for r in range(1,rank):
	    indVarName = 'i_r_%d' % (r)
	    newLoop = buildSimpleForLoop(indVarName, numNodesPerEle)
	    loop.append(newLoop)
	    loop = newLoop
	return outerLoop

def buildLoopNest(form):
    LNB = LoopNestBuilder()
    return LNB.build(form)

def buildKernel(form):
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
# Eventually these need to be set by the callee of the code generator
numNodesPerEle = 3
