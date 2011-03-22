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


