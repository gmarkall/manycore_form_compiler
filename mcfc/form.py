"""form.py - contains the code shared by different form backends, e.g.
cudaform.py, op2form.py, etc."""

# MCFC libs
from codegeneration import *
# UFL libs
from ufl.algorithms.transformations import Transformer

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

# Code indices represent induction variables in a loop. A list of CodeIndexes is
# supplied to buildOffset, in order for it to build the computation of a
# subscript.

class CodeIndex:

    def __init__(self, count=None):
        self._count = count

class RankIndex(CodeIndex):
    
    def extent(self):
        return Literal(_impl.numNodesPerEle)

    def name(self):
        return _impl.basisInductionVariable(self._count)

class ElementIndex(CodeIndex):

    def extent(self):
        return _impl.numElements

    def name(self):
        return _impl.eleInductionVariable()

class GaussIndex(CodeIndex):

    def extent(self):
        return Literal(_impl.numGaussPoints)

    def name(self):
        return _impl.gaussInductionVariable()

class DimIndex(CodeIndex):

    def extent(self):
        return Literal(_impl.numDimensions)

    def name(self):
        return _impl.dimInductionVariable(self._count)

def buildOffset(indices):
    """Given a list of indices, return an AST that computes
    the offset into an array using those indices. The order is
    important."""

    # Start our expression with the first index
    name = indices[0].name()
    offset = Variable(name)
    
    # Compute the expression for all indices
    for v in range(1,len(indices)):
        subindices = indices[:v]
	name = indices[v].name()
	expr = Variable(name)
	
	# Find the correct offset for this index
	for u in range(len(subindices)):
	    multiplier = subindices[u].extent()
	    expr = MultiplyOp(multiplier, expr)
	offset = AddOp(offset, expr)
    
    return offset

# Global variables for code generation.
# Eventually these need to be set by the caller of the code generator
numNodesPerEle = 3
numDimensions = 2
numGaussPoints = 6

# Implementation registation

def registerImplementation(implementation):
    global _impl
    _impl = implementation

_impl = None
