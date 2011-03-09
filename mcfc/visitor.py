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

class UFLVisitor:

    def __init__(self, order):
        self._order = order

    def traverse(self, tree):
        if self._order == preOrder:
	    self.visit(tree)

	for op in tree.operands():
	    self.traverse(op)
	
	if self._order == postOrder:
	    self.visit(tree)

    def visit(self, tree):
        meth = getattr(self, "_"+tree.__class__.__name__)
	meth(tree)

    def _Product(self, tree):
        print "Product"

    def _Sum(self, tree):
        print "Sum"

    def _SpatialDerivative(self, tree):
        print "SpatialDerivative"

    def _ComponentTensor(self, tree):
        print "ComponentTensor"

    def _Indexed(self, tree):
        print "Indexed"

    def _IndexSum(self, tree):
        print "IndexSum"

    def _Argument(self, tree):
        print "Argument"

    def _MultiIndex(self, tree):
        print "MultiIndex"

    def _IntValue(self, tree):
        print "IntValue"

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
