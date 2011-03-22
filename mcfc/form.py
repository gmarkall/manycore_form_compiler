"""form.py - contains the code shared by different form backends, e.g.
cudaform.py, op2form.py, etc."""

# MCFC libs
from codegeneration import *
# UFL libs
from ufl.algorithms.transformations import Transformer

class ExpressionBuilder(Transformer):

    def build(self, tree):
        self._exprStack = []
	# When we pass through the first IndexSum, this will get incremented
	# to 0, which is the count of the first dim index
	self._indexSumDepth = -1 
        self.visit(tree)

	expr = self._exprStack.pop()

	if len(self._exprStack) is not 0:
	    raise RuntimeError("Expression stack not empty.")

        # Everything needs to be multiplied by detwei
	indices = self.subscript_detwei()
	offset = buildOffset(indices)
	detwei = Variable("detwei")
	detweiExpr = Subscript(detwei, offset)
	expr = MultiplyOp(expr, detweiExpr)

	return expr

    def component_tensor(self, tree, *ops):
        pass

    def indexed(self, tree, *ops):
        pass

    # We need to keep track of how many IndexSums we passed through
    # so that we know which dim index we're dealing with.
    def index_sum(self, tree):
        summand, indices = tree.operands()
	self._indexSumDepth = self._indexSumDepth + 1
        self.visit(summand)
	self._indexSumDepth = self._indexSumDepth - 1

    def constant_value(self, tree):
        value = Literal(tree.value())
	self._exprStack.append(value)

    def sum(self, tree, *ops):
	rhs = self._exprStack.pop()
        lhs = self._exprStack.pop()
	add = AddOp(lhs, rhs)
	self._exprStack.append(add)

    def product(self, tree, *ops):
	rhs = self._exprStack.pop()
        lhs = self._exprStack.pop()
	multiply = MultiplyOp(lhs, rhs)
	self._exprStack.append(multiply)

    def spatial_derivative(self, tree):
        name = buildSpatialDerivativeName(tree)
	base = Variable(name)

	depth = self._indexSumDepth
	indices = self.subscript(tree, depth)
	offset = buildOffset(indices)
	spatialDerivExpr = Subscript(base, offset)
	self._exprStack.append(spatialDerivExpr)
 
    def argument(self, tree):
        name = buildArgumentName(tree)
        base = Variable(name)

        indices = self.subscript(tree)
	offset = buildOffset(indices)
	argExpr = Subscript(base, offset)
        self._exprStack.append(argExpr)

    def coefficient(self, tree):
	coeffExpr = self.buildCoeffQuadratureAccessor(tree)
	self._exprStack.append(coeffExpr)

    def buildCoeffQuadratureAccessor(self, coeff):
	name = buildCoefficientQuadName(coeff)
	base = Variable(name)
	
	indices = self.subscript_CoeffQuadrature(coeff)
	offset = buildOffset(indices)

	coeffExpr = Subscript(base, offset)
	return coeffExpr

    def buildLocalTensorAccessor(self, form):
	indices = self.subscript_LocalTensor(form)
	offset = buildOffset(indices)
	
	# Subscript the local tensor variable
	expr = Subscript(localTensor, offset)
	return expr

    def subscript_LocalTensor(self, form):
        raise NotImplementedError("You're supposed to implement subscript_LocalTensor()!")

    def subscript_CoeffQuadrature(self, coeff):
        raise NotImplementedError("You're supposed to implement subscript_CoeffQuadrature()!")

    def subscript(self, tree):
        raise NotImplementedError("You're supposed to implement subscript()!")

    def subscript_detwei(self, tree):
        raise NotImplementedError("You're supposed to implement subscript_detwei()!")

class QuadratureExpressionBuilder:

    def build(self, tree):
        indices = self.subscript(tree)
	offset = buildOffset(indices)
	coeffAtBasis = Variable(buildCoefficientName(tree))
	expr = Subscript(coeffAtBasis, offset)
	return expr

    def subscript(self, tree):
        raise NotImplementedError("You're supposed to implement subscript()!")

class FormBackend:

    def __init__(self):
        self._expressionBuilder = None
	self._quadratureExpressionBuilder = None

    def buildExpression(self, form, tree):
	"Build the expression represented by the subtree tree of form."
	# Build the rhs expression
	rhs = self._expressionBuilder.build(tree)

	# Assign expression to the local tensor value
	lhs = self._expressionBuilder.buildLocalTensorAccessor(form)
	expr = PlusAssignmentOp(lhs, rhs)

	return expr
        
    def buildQuadratureExpression(self, coeff):
	rhs = self._quadratureExpressionBuilder.build(coeff)

	lhs = self._expressionBuilder.buildCoeffQuadratureAccessor(coeff)
	expr = PlusAssignmentOp(lhs, rhs)
	
	return expr
    
    def buildLocalTensorInitialiser(self, form):
	lhs = self._expressionBuilder.buildLocalTensorAccessor(form)
	rhs = Literal(0.0)
	initialiser = AssignmentOp(lhs, rhs)
	return initialiser

    def buildCoeffQuadratureInitialiser(self, coeff):
	accessor = self._expressionBuilder.buildCoeffQuadratureAccessor(coeff)
	initialiser = AssignmentOp(accessor, Literal(0.0))
	return initialiser

    def buildCoeffQuadDeclarations(self, form):
	form_data = form.form_data()
	coefficients = form_data.coefficients
	declarations = []

	for coeff in coefficients:
	    name = buildCoefficientQuadName(coeff)
	    rank = coeff.rank()
	    length = numGaussPoints * pow(numDimensions, rank)
	    t = Array(Real(), Literal(length))
	    var = Variable(name, t)
	    decl = Declaration(var)
	    declarations.append(decl)

	return declarations

    def compile(self, form):
        raise NotImplementedError("You're supposed to implement compile()!")

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
        return Literal(numNodesPerEle)

    def name(self):
        return basisInductionVariable(self._count)

class GaussIndex(CodeIndex):

    def extent(self):
        return Literal(numGaussPoints)

    def name(self):
        return gaussInductionVariable()

class DimIndex(CodeIndex):

    def extent(self):
        return Literal(numDimensions)

    def name(self):
        return dimInductionVariable(self._count)

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

# Name builders

def buildArgumentName(tree):
    element = tree.element()
    name = element.shortstr()
    return name

def buildSpatialDerivativeName(tree):
    argument = tree.operands()[0]
    argName = buildArgumentName(argument)
    spatialDerivName = 'd_%s' % (argName)
    return spatialDerivName

def buildCoefficientName(tree):
    count = tree.count()
    name = 'c%d' % (count)
    return name

def buildCoefficientQuadName(tree):
    count = tree.count()
    name = 'c_q%d' %(count)
    return name

# Names of induction variables

def gaussInductionVariable():
    return "i_g"

def basisInductionVariable(count):
    name = "i_r_%d" % (count)
    return name

def dimInductionVariable(count):
    name = "i_d_%d" % (count)
    return name

# Global variables for code generation.
# Eventually these need to be set by the caller of the code generator
numNodesPerEle = 3
numDimensions = 2
numGaussPoints = 6

# Variables used globally

detwei = Variable("detwei", Pointer(Real()) )
timestep = Variable("dt", Real() )
localTensor = Variable("localTensor", Pointer(Real()) )
