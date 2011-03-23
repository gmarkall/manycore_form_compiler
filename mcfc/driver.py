import cudaform
import cudaassembler
from visitor import *

import ufl.form

class FormFinder(AntlrVisitor):

    def __init__(self):
        AntlrVisitor.__init__(self, preOrder)

    def find(self, tree):
        self._forms = []
	self._count = 0
	self.traverse(tree)
	return self._forms

    def visit(self, tree):
        label = str(tree)

	if label == '=':
	    lhs = tree.getChild(0)
	    rhs = tree.getChild(1)
	    if str(rhs) == 'Form':
	        self._forms.append((str(lhs), self._count))
		self._count = self._count + 1
    
    def pop(self):
        pass

def drive(ast, uflObjects, fd):

    formBackend = cudaform.CudaFormBackend()
    assemblerBackend = cudaassembler.CudaAssemblerBackend()
    formFinder = FormFinder()

    # Build headers
    headers = assemblerBackend.buildHeadersAndGlobals(ast, uflObjects)
    print >>fd, headers.unparse()
    print >>fd

    # Build forms
    forms = formFinder.find(ast)
    for form, count in forms:
        o = uflObjects[form]
	name = form + '_' + str(count)
	code = formBackend.compile(name, o)
	print >>fd, code.unparse()
	print >>fd

    # Build assembler
    state = assemblerBackend.buildState()
    initialiser = assemblerBackend.buildInitialiser(ast, uflObjects)
    finaliser = assemblerBackend.buildFinaliser(ast, uflObjects)
    print >>fd, state.unparse()
    print >>fd
    print >>fd, initialiser.unparse()
    print >>fd
    print >>fd, finaliser.unparse()
