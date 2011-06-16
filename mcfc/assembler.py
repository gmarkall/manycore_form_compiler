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

from ast import NodeVisitor
from ufl.coefficient import Coefficient

# Placeholder; will probably fill with things later on.
# Should be inherited by all assembler implementations.
class AssemblerBackend:

    def compile(self, ast, uflObjects):
        raise NotImplementedError("You're supposed to implement compile()!")

class AccessedFieldFinder(NodeVisitor):

    def find(self, tree):
        self._fields = []
	self.visit(tree)
	return self._fields

    def visit_Assign(self, tree):
	rhs = tree.value
	if len(tree.targets)==1:
	    try:
	        # subscript -> attribute -> name
	        objname = rhs.value.value.id
		# subscript -> attribute -> string
		objmember = rhs.value.attr
		if objname == "state":
		    if objmember == "scalar_fields":
		        rank = 0
		    if objmember == "vector_fields":
		        rank = 1
		    if objmember == "tensor_fields":
		        rank = 2
		    fieldname = rhs.slice.value.s
		    self._fields.append((rank, fieldname))
	    except AttributeError:
	        # This is not a field access, so no action necessary.
		pass
	else:
	    raise NotImplementedError("Tuple assignment needs implementing.")

def findAccessedFields(tree):
    AFF = AccessedFieldFinder()
    return AFF.find(tree)

class SolveResultFinder(NodeVisitor):

    def find(self, tree):
        self._results = []
	self.visit(tree)
	return self._results

    def visit_Assign(self, tree):
        rhs = tree.value
	if len(tree.targets)==1:
	    try:
	        func = rhs.func.id
		if func == "solve":
		    result = tree.targets[0].id
		    self._results.append(result)
	    except AttributeError:
	        # This is not a call to a solve, so no action is required anywya
	        pass
	else:
	    raise NotImplementedError("Tuple assignment not implemented")

def findSolveResults(tree):
    SRF = SolveResultFinder()
    return SRF.find(tree)

class SolveFinder(NodeVisitor):
    """ Traverses an antlr tree and returns the assignment node of
    solves, rather than the solve node itself. This way we can get
    to the target as well as the solve node."""

    def find(self, tree):
        self._solves = []
        self.visit(tree)
        return self._solves
    
    def visit_Assign(self, tree):
        rhs = tree.value
	if len(tree.targets)==1:
	    try:
	        func = rhs.func.id
		if func == "solve":
		    self._solves.append(tree)
	    except AttributeError:
	        # This is not a call to a solve, so no action is required anywya
	        pass
	else:
	    raise NotImplementedError("Tuple assignment not implemented.")

def findSolves(tree):
    SF = SolveFinder()
    return SF.find(tree)

class FieldVarFinder(NodeVisitor):
    
    def find(self, ast, name):
        self._name = name
        self._fieldVar = None
        self.visit(ast)
        return self._fieldVar

    def visit_Assign(self, tree):
        if len(tree.targets) == 1:
	    target = tree.targets[0]
	    try:
	        var = target.id
	    except AttributeError:
	        # If this happens, the LHS is not what we're looking for.
		return
	    if var == self._name:
	        try:
		    self._fieldVar = tree.value.args[0].id
	        except AttributeError:
		    # If we got here, then the RHS was not as expected
		    raise RuntimeError("Unexpected RHS for coefficient %s" % self._name)
	else:
	    raise NotImplementedError("Tuple assignment not implemented.")

class FieldNameFinder(NodeVisitor):

    def find(self, ast, var):
        self._var = var
        self._field = None
        self.visit(ast)
        return self._field

    def visit_Assign(self, tree):
        if len(tree.targets) == 1:
	    target = tree.targets[0]
	    try:
	        var = target.id
            except AttributeError:
	        # If this happens, the LHS is not what we're looking for.
		return
	    try:
	        if var == self._var:
	            self._field = tree.value.slice.value.s
            except AttributeError:
		# If we got here, then the RHS was not as expected
		raise RuntimeError("Unexpected RHS for field var %s" % self._var)
	else:
	    raise NotImplementedError("Tuple assignment not implemented.")
	   
def findCoefficientName(uflObjects, coeff):
    seeking = coeff.count()
    for key, value in uflObjects.items():
        if isinstance(value, Coefficient):
	    count = value.count()
	    if count == seeking:
	        return key
    print "Warning: coefficient not found."

def findFieldFromCoefficient(ast, uflObjects, coeff):
    FVF = FieldVarFinder()
    FNF = FieldNameFinder()
    var = FVF.find(ast, findCoefficientName(uflObjects, coeff))
    return FNF.find(ast, var)

class ReturnedFieldFinder(NodeVisitor):
    """Return a list of the fields that need to be returned to the host. These
    are pairs (hostField, GPUField), where hostField is the name of the field
    that will be overwritted with data currently stored in GPUField."""

    def find(self, ast):
       self._returnFields = []
       self.visit(ast)
       return self._returnFields

    def visit_Assign(self, tree):
	if len(tree.targets) == 1:
	    try:
	        lhs = tree.targets[0]
                rhs = tree.value
	        # subscript -> attribute -> name
		objname = lhs.value.value.id
		# subscript -> attribute -> string
		objmember = lhs.value.attr
		fieldholders = ['scalar_fields', 'vector_fields', 'tensor_fields']
		if objname == "state" and objmember in fieldholders:
		    hostField = lhs.slice.value.s
		    GPUField  = rhs.id
		    self._returnFields.append((hostField, GPUField))
            except AttributeError:
	        # This is not a returning of a field, so no action required.
	        pass
	else:
	    raise NotImplementedError("Tuple assignment not implemented.")

def findReturnedFields(ast):
    RFF = ReturnedFieldFinder()
    return RFF.find(ast)

# vim:sw=4:ts=4:sts=4:et
