# This file is part of the Manycore Form Compiler.
#
# The Manycore Form Compiler is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
# 
# The Manycore Form Compiler is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# more details.
# 
# You should have received a copy of the GNU General Public License along with
# the Manycore Form Compiler.  If not, see <http://www.gnu.org/licenses>
#
# Copyright (c) 2011, Graham Markall <grm08@doc.ic.ac.uk> and others. Please see
# the AUTHORS file in the main source directory for a full list of copyright
# holders.


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
