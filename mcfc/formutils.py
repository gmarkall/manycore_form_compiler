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

"formutils.py - contains helper classes and methods used by the form backends"

# UFL libs
from ufl.argument import Argument
from ufl.coefficient import Coefficient
from ufl.common import Counted
from ufl.expr import Expr
from ufl.form import Form
from ufl.algorithms.transformations import Transformer
from ufl.differentiation import SpatialDerivative
from ufl.finiteelement import FiniteElement, VectorElement, TensorElement
from ufl.classes import all_ufl_classes
from ufl.common import camel2underscore as _camel2underscore

# FFC libs
from ffc.fiatinterface import create_element
from ffc.mixedelement import MixedElement as FFCMixedElement

# MCFC libs
from codegeneration import *
from utilities import uniqify

# Extra AST nodes, and a little bit of trickery to make them work with
# Transformer objects.

class SubExpr(Expr, Counted):
    _globalcount = 0

    def __init__(self, o, count=None):
        Expr.__init__(self)
        Counted.__init__(self, count=count, countedclass=SubExpr)
        self._operand = o
        self._repr = "SubTree(%s)" % repr(o)
        self._shape = o.shape()
        self._str = str(o)

    def reconstruct(self, *ops):
        return SubExpr(ops)

    def operands(self):
        return [self._operand]

    def shape(self):
        return self._shape

    def evaluate(self, x, mapping, component, index_values):
        return self._operand.evaluate(x, mapping, component, index_values)

    def free_indices(self):
        return self._operand.free_indices()

    def index_dimensions(self):
        return self._operand.index_dimensions()

    def __repr__(self):
        return self._repr

    def __str__(self):
        return self._str


# Start the classid count from where the last UFL class counter left off.
_classid_count = len(all_ufl_classes)

all_our_ufl_classes = set([SubExpr])

for _i, _c in enumerate(all_our_ufl_classes):
    _c._classid = _i + _classid_count
    _c._uflclass = _c
    _c._handlername = _camel2underscore(_c.__name__)


# A transformer that works with our extra UFL classes.

class UserDefinedClassTransformer(Transformer):
    """A Transformer that also works on the UFL classes that are defined in
    all_our_ufl_classes"""

    def __init__(self, variable_cache=None):
        # Get the cache data first so we know if this is the first instantiation
        # of this type
        cache_data = Transformer._handlers_cache.get(type(self))
        # The Transformer then handles initialisation for the UFL classes
        Transformer.__init__(self, variable_cache)
        # Now we may need to handle initialisation for our AST nodes
        if not cache_data:
            # Get the handler_cache contents that Transformer just created
            cache_data = Transformer._handlers_cache.get(type(self))
            # Make some space for handlers for our classes
            extra_cache_data = [None]*len(all_our_ufl_classes)
            cache_data.extend(extra_cache_data)
            # For all our UFL classes
            for classobject in all_our_ufl_classes:
                for c in classobject.mro():
                    name = c._handlername 
                    function = getattr(self, name, None)
                    if function:
                        cache_data[classobject._classid] = name, is_post_handler(function)
                        break
            Transformer._handlers_cache[type(self)] = cache_data

        self._handlers = [(getattr(self, name), post) for (name, post) in cache_data]


class CoefficientUseFinder(UserDefinedClassTransformer):
    """Finds the nodes that 'use' a coefficient. This is either a Coefficient
    itself, or a SpatialDerivative that has a Coefficient as its operand"""

    def find(self, tree):
        # We keep coefficients and spatial derivatives in separate lists
        # because we need separate criteria to uniqify the lists.
        self._coefficients = []
        self._spatialDerivatives = []

        self.visit(tree)

        # Coefficients define __eq__ and __hash__ so the straight uniqify works.
        # For spatial derivatives, we need to compare the coefficients.
        coefficients = uniqify(self._coefficients)
        spatialDerivatives = uniqify(self._spatialDerivatives, lambda x: x.operands()[0])

        return coefficients, spatialDerivatives

    # Most expressions are uninteresting.
    def expr(self, tree, *ops):
        pass

    def argument(self, tree):
        pass

    def spatial_derivative(self, tree):
        subject = tree.operands()[0]
        if isinstance(subject, Coefficient):
            self._spatialDerivatives.append(tree)

    def coefficient(self, tree):
        self._coefficients.append(tree)

class IndexSumIndexFinder(UserDefinedClassTransformer):
    "Find the count and extent of indices reduced by an IndexSum in a form."

    def find(self, tree):
        self._indices = []
        self.visit(tree)
        return self._indices

    def index_sum(self, tree):

        summand, mi = tree.operands()
        indices = mi.index_dimensions()

        for c, d in indices.items():
            self._indices.append(buildDimIndex(c.count(), d))

        self.visit(summand)

    # We don't care about any other node.
    def expr(self, tree, *ops):
        pass

def indexSumIndices(tree):
    ISIF = IndexSumIndexFinder()
    return ISIF.find(tree)

class Partitioner(UserDefinedClassTransformer):
    """Partitions the expression up so that each partition fits inside
    strictly on loop in the local assembly loop nest."""

    def binary_op(self, tree):
        ops = tree.operands()
        lInd = indexSumIndices(ops[0])
        rInd = indexSumIndices(ops[1])
        # If both sides have the same nesting level:
        if lInd == rInd:
            return
        else:
            ops = [self.visit(ops[0]), self.visit(ops[1])]
            return SubTree(tree.__class__(ops))

    sum = binary_op
    product = binary_op

def partition(tree):
    part = Partitioner()
    return part.visit(tree)

def buildLoopNest(scope, indices):
    """Build a loop nest using the given indices in the given scope. Reuse
    existing loops that use the same indices."""
    for i in indices:
        # Find a loop that uses the current index
        # FIXME: We don't have a test case using that
        loop = scope.find(lambda x: isinstance(x, ForLoop) \
                and x._inc._expr == i)
        # Build the loop if we haven't already found one with that index
        if not loop:
            loop = buildIndexForLoop(i)
            scope.append(loop)
        scope = loop.body()
    return scope

# Code indices represent induction variables in a loop. A list of CodeIndexes is
# supplied to buildSubscript, in order for it to build the computation of a
# subscript.

class CodeIndex:

    def __init__(self, extent, count=None):
        self._extent = extent
        self._count = count

    def extent(self):
        return Literal(self._extent)

class BasisIndex(CodeIndex):

    def name(self):
        return "i_r_%d" % (self._count)

class GaussIndex(CodeIndex):

    def name(self):
        return "i_g"

class DimIndex(CodeIndex):

    def name(self):
        return "i_d_%d" % (self._count)

class ConstIndex(CodeIndex):

    def name(self):
        return str(self._count)

# Index builders

def extract_element(expr):
    "Extract the element of a UFL expression."
    if isinstance(expr, SpatialDerivative):
        expr = expr.operands()[0]
    return extractCoordinates(expr).element()

def extract_subelement(expr):
    """Extract the scalar element of a UFL expression (i.e. for a vector or
    tensor element, extract the finite element it is composed from)."""
    e = extract_element(expr)
    if e.num_sub_elements() > 0:
        return e.sub_elements()[0]
    return e

def buildBasisIndex(count, e):
    "Build index for a loop over basis function values."
    return BasisIndex(numBasisFunctions(e), count)

def buildDimIndex(count, e):
    "Build index for a loop over spatial dimensions."
    # If passed an int, interpret it as the spatial dimension
    if isinstance(e, int):
        dim = e
    # Otherwise we assume we're passed an element and extract the dimension
    else:
        dim = e.cell().topological_dimension()
    return DimIndex(dim, count)

def buildConstDimIndex(count):
    "Build literal subscript for a loop over spatial dimensions."
    # FIXME: We pass a constant extent of 1 since the actual extent is
    # non-trivial to extracted from the context and is not required when the
    # index is used to subscript a multi-dimensional array
    return ConstIndex(1, count)

def buildGaussIndex(n):
    "Build index for a Gauss quadrature loop."
    return GaussIndex(n)

# Name builders

def safe_shortstr(name):
    return name[:name.find('(')]

def buildArgumentName(tree):
    element = tree.element()
    if element.num_sub_elements() is not 0:
        if element.family() == 'Mixed':
            raise NotImplementedError("I can't digest mixed elements. They make me feel funny.")
        else:
            # Sub elements are all the same
            sub_elements = element.sub_elements()
            element = sub_elements[0]

    return safe_shortstr(element.shortstr())

def buildSpatialDerivativeName(tree):
    operand = tree.operands()[0]
    if isinstance(operand, Argument):
        name = buildArgumentName(operand)
    elif isinstance(operand, Coefficient):
        name = buildCoefficientQuadName(operand)
    else:
        cls = operand.__class__.__name__
        raise NotImplementedError("Unsupported SpatialDerivative of " + cls)
    spatialDerivName = 'd_%s' % (name)
    return spatialDerivName

def buildCoefficientName(tree):
    count = tree.count()
    name = 'c%d' % (count)
    return name

def buildCoefficientQuadName(tree):
    count = tree.count()
    name = 'c_q%d' %(count)
    return name

def buildVectorArgumentName(tree):
    return buildArgumentName(tree) + "_v"

def buildVectorSpatialDerivativeName(tree):
    return buildSpatialDerivativeName(tree) + "_v"

def buildTensorArgumentName(tree):
    return buildArgumentName(tree) + "_t"

def buildTensorSpatialDerivativeName(tree):
    return buildSpatialDerivativeName(tree) + "_t"

def isJacobian(coeff):
    "Detect whether a coefficient represents the Jacobian."
    return isinstance(coeff.element().quadrature_scheme(), Coefficient)

def extractCoordinates(coeff):
    """Extract the coordinate coefficient from the Jacobian coefficient. Does
    nothing if the coefficient doesn't represent the Jacobian."""
    if isJacobian(coeff):
        return coeff.element().quadrature_scheme()
    return coeff

def numBasisFunctions(e):
    """Return the number of basis functions. e can be a form or an element -
    if e is a form, the element from the test function is used."""
    if isinstance(e, Form):
        # Use the element from the first argument, which should be the TestFunction
        e = e.form_data().arguments[0].element()
    element = create_element(e)
    if isinstance(element, FFCMixedElement):
        return len(element.entity_dofs()) * element.num_components()
    else:
        return element.get_nodal_basis().get_num_members()

# vim:sw=4:ts=4:sts=4:et
