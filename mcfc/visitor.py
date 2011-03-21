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
