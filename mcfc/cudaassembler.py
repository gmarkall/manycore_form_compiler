"""This module generates the code that extracts the relevant fields from
Fluidity state, transfers it to the GPU, and the run_model_ function that
executes the model for one timestep, by calling the kernels generated by
cudaform.py, and the necessary solves."""

from visitor import *
from codegeneration import *

def buildStateType():
    return Pointer(Class('StateHolder'))

def buildState():
    t = buildStateType()
    state = Variable('state', t)
    decl = Declaration(state)
    return decl

def buildInitialiser(AST):

    func = FunctionDefinition(Void(), 'initialise_gpu_')
    func.setExternC(True)

    # Call the state constructor
    state = Variable('state')
    newState = New(Class('StateHolder'))
    construct = AssignmentOp(state, newState)
    func.append(construct)
    
    # Call the state initialiser
    call = FunctionCall('initialise')
    arrow = ArrowOp(state, call)
    func.append(arrow)

    # Extract accessed fields
    accessedFields = findAccessedFields(AST)
    for field in accessedFields:
        fieldString = '"' + field + '"'
        params = ExpressionList([Literal(fieldString)])
        call = FunctionCall('extractField',params)
	arrow = ArrowOp(state, call)
	func.append(arrow)

    # Allocate memory and transfer to GPU
    call = FunctionCall('allocateAllGPUMemory')
    arrow = ArrowOp(state, call)
    func.append(arrow)

    call = FunctionCall('transferAllFields')
    arrow = ArrowOp(state, call)
    func.append(arrow)
    
    # Insert temporary fields into state
    solveResultFields = findSolveResults(AST)
    for field in solveResultFields:
        fieldString = '"' + field + '"'
	params = ExpressionList([Literal(fieldString)])
	call = FunctionCall('insertTemporaryField',params)
	arrow = ArrowOp(state, call)
	func.append(arrow)

    ########################
    #### Get num_ele, num_nodes etc
    ##########################

    ##########################
    ## do mallocs
    ##########################

    return func

class AccessedFieldFinder(AntlrVisitor):

    def __init__(self):
        AntlrVisitor.__init__(self, preOrder)

    def find(self, tree):
        self._fields = []
	self.traverse(tree)
	return self._fields

    def visit(self, tree):
        label = str(tree)

	if label == '=':
	    rhs = tree.getChild(1)
	    if str(rhs) in ['scalar_fields', 'vector_fields', 'tensor_fields']:
		field = str(rhs.getChild(0))
		# Strip off the quotes
		field = field[1:-1]
		self._fields.append(field)

    def pop(self):
        pass

def findAccessedFields(tree):
    AFF = AccessedFieldFinder()
    return AFF.find(tree)

class SolveResultFinder(AntlrVisitor):

    def __init__(self):
        AntlrVisitor.__init__(self, preOrder)

    def find(self, tree):
        self._results = []
	self.traverse(tree)
	return self._results

    def visit(self, tree):
        label = str(tree)

	if label == '=':
	    rhs = tree.getChild(1)
	    if str(rhs) == 'solve':
	        result = str(tree.getChild(0))
		self._results.append(result)

    def pop(self):
        pass

def findSolveResults(tree):
    SRF = SolveResultFinder()
    return SRF.find(tree)

