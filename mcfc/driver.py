import cudaform
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

def drive(ast, uflObjects):

    formBackend = cudaform.CudaFormBackend()
    formFinder = FormFinder()

    forms = formFinder.find(ast)

    for form, count in forms:
        o = uflObjects[form]
	name = form + '_' + str(count)
	ast = formBackend.compile(name, o)
	print ast.unparse()
	print
