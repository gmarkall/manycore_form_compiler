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

def buildKernel(tree):
    t = Void()
    name = "kernel" # Fix later
    params = buildParameterList(tree)
    expression = buildExpression(tree)
    statements = [expression]
    body = Scope(statements)
    kernel = FunctionDefinition(t, name, params, body)
    return kernel
