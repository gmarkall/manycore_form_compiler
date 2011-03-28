from visitor import *

import ufl.form

class FormFinder(AntlrVisitor):

    def __init__(self):
        AntlrVisitor.__init__(self, preOrder)

    def find(self, tree):
        self._forms = []
	self.traverse(tree)
	return self._forms

    def visit(self, tree):
        label = str(tree)

	if label == '=':
	    lhs = tree.getChild(0)
	    rhs = tree.getChild(1)
	    if str(rhs) == 'Form':
	        self._forms.append(str(lhs))
    
    def pop(self):
        pass

class Driver:

    def drive(self, ast, uflObjects, fd):

        formFinder = FormFinder()

        # Build assembler
        definitions, declarations = self._assemblerBackend.compile(ast, uflObjects)
        # Unparse assembler defintions (headers)
        print >>fd, definitions.unparse()
        print >>fd

        # Build forms
        forms = formFinder.find(ast)
        for form in forms:
            o = uflObjects[form]
            name = form
            code = self._formBackend.compile(name, o)
            print >>fd, code.unparse()
            print >>fd

        # Unparse assembler declarations (body)
        print >>fd, declarations.unparse()
        print >>fd
