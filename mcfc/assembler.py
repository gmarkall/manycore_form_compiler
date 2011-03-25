from visitor import *

# Placeholder; will probably fill with things later on.
# Should be inherited by all assembler implementations.
class AssemblerBackend:
    pass

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

class SolveFinder(AntlrVisitor):
    """ Traverses an antlr tree and returns the assignment node of
    solves, rather than the solve node itself. This way we can get
    to the target as well as the solve node."""

    def __init__(self):
        AntlrVisitor.__init__(self, preOrder)

    def find(self, tree):
        self._solves = []
	self.traverse(tree)
	return self._solves

    def visit(self, tree):
        label = str(tree)

	if label == '=':
	    rhs = tree.getChild(1)
	    if str(rhs) == 'solve':
	        self._solves.append(tree)
    
    def pop(self):
        pass

def findSolves(tree):
    SF = SolveFinder()
    return SF.find(tree)

class CoefficientFieldFinder(AntlrVisitor):
    """Given a coefficient, this class traverses the
    AST and finds the name of the field it came from"""

    def __init__(self):
        AntlrVisitor.__init__(self, preOrder)

    def find(self, ast, coeff):
        self._count = str(coeff.count())
	self._field = None
	self.traverse(ast)
	return self._field

    def visit(self, tree):
        label = str(tree)

	if label == 'Coefficient':
	    count = str(tree.getChild(1))
	    if count == self._count:
	        # Found the correct Field. However, we need to make sure this
		# was a version of the field alone on the right-hand side of 
		# an assignment.
		field = tree.getParent()
		fieldParent = field.getParent()
		if str(fieldParent) == '=':
		    self._field = str(field)

    def pop(self):
        pass

def findFieldFromCoefficient(ast, coeff):
    CFF = CoefficientFieldFinder()
    return CFF.find(ast, coeff)
