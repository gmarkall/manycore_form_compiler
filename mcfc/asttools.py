import visitor

def findForms(tree):
    f = _Finder('Form')
    return f.search(tree)

class _Finder(visitor.AntlrVisitor):

    def __init__(self,node):
        visitor.AntlrVisitor.__init__(self, visitor.preOrder)
        self._forms = []
	self._node = node

    def search(self, tree):
        self.traverse(tree);
	return self._forms

    def visit(self, tree):
        if str(tree) == self._node:
	    self._forms.append(tree)

    def pop(self):
        pass


