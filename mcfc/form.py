from ufl.algorithms.transformations import Transformer

"""form.py - contains the code shared by different form backends, e.g.
cudaform.py, op2form.py, etc."""

class IndexSumCounter(Transformer):
    "Count how many IndexSums are nested inside a tree."

    def count(self, tree):
        self._indexSumDepth = 0
        self._maxIndexSumDepth = 0
        self.visit(tree)
	return self._maxIndexSumDepth

    def index_sum(self, tree):
	
	summand, indices = tree.operands()

        self._indexSumDepth = self._indexSumDepth + 1
	if self._indexSumDepth > self._maxIndexSumDepth:
	    self._maxIndexSumDepth = self._indexSumDepth

	self.visit(summand)

	self._indexSumDepth = self._indexSumDepth - 1

    # We don't care about any other node.
    def expr(self, tree, *ops):
        pass

    def terminal(self, tree):
        pass

class Partitioner(Transformer):
    """Partitions the expression up so that each partition fits inside
    strictly on loop in the local assembly loop nest.
    Returns a list of the partitions, and their depth (starting from
    inside the quadrature loop)."""

    def partition(self, tree):
        self._partitions = []
	self._ISC = IndexSumCounter()
	self.visit(tree)
	return self._partitions

    def sum(self, tree):
        ops = tree.operands()
	lDepth = self._ISC.count(ops[0])
	rDepth = self._ISC.count(ops[1])
	# If both sides have the same nesting level:
	if lDepth == rDepth:
	    self._partitions.append((tree,lDepth))
	    return
	else:
	    self.visit(ops[0])
	    self.visit(ops[1])
    
    # If it's not a sum, then there shouldn't be any partitioning
    # of the tree anyway.
    def expr(self, tree):
        depth = self._ISC.count(tree)
        self._partitions.append((tree,depth))

def findPartitions(tree):
    part = Partitioner()
    return part.partition(tree)


