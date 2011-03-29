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



"""visitor.py - Provides an object for traversing ANTLR ASTs."""

class AntlrVisitor:

    def __init__(self, order):
        self._order = order

    def traverse(self, tree):
        if self._order == preOrder:
	    self.visit(tree)

	for i in range(tree.getChildCount()):
	    child = tree.getChild(i)
	    self.traverse(child)
	
	if self._order == postOrder:
	    self.visit(tree)

	self.pop()

class TraversalOrder:

    def __init__(self, name):
        self._name = name
	self._repr = "TraversalOrder('" + name + "')"

    def __repr__(self):
        return self._repr

    def __eq__(self, other):
        return (self._name == other._name)

preOrder = TraversalOrder("preOrder")
postOrder = TraversalOrder("postOrder")
