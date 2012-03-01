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

from form import *
from symbolicvalue import SymbolicValue
from ufl.argument import TrialFunction
from ufl.coefficient import Coefficient
from ufl.common import Stack
from ufl.differentiation import SpatialDerivative
from ufl.finiteelement import FiniteElement, VectorElement, TensorElement
from ufl.indexing import Index, FixedIndex

def buildMultiArraySubscript(variable, indices):
    """Given a list of indices, return an AST of the variable subscripted by
    the indices as a multidimensional array. The index order is important."""

    for i in indices:
        variable = Subscript(variable, Variable(i.name()))

    return variable

class ExpressionBuilder(UserDefinedClassTransformer):

    def __init__(self, formBackend):
        UserDefinedClassTransformer.__init__(self)
        self._formBackend = formBackend

    def build(self, tree):
        "Build the rhs for evaluating an expression tree."
        self._subExprStack = []
        self._indexStack = Stack()
        self._sumIndexStack = Stack()
        expr = self.visit(tree)

        assert len(self._indexStack) == 0, "Index stack not empty."

        return expr, self._subExprStack

    def subscript(self, tree):
        meth = getattr(self, "subscript_"+tree.__class__.__name__)
        return meth(tree)

    def subscript_Argument(self, tree):
        # Build the subscript based on the argument count
        indices = []
        # For vector valued function space we need to add an index
        # over dimensions
        if len(self._indexStack) > 0:
            indices.extend(self._indexStack.peek())
        indices += [buildBasisIndex(tree.count(), tree.element()),
                    buildGaussIndex(self._formBackend.numGaussPoints)]
        return indices

    def subscript_Coefficient(self, coeff):
        # Build the subscript based on the rank
        indices = [buildGaussIndex(self._formBackend.numGaussPoints)]

        rank = coeff.rank()

        if isinstance(coeff, SpatialDerivative):
            rank = rank + 1
        if rank > 0:
            dimIndices = self._indexStack.peek()
            if len(dimIndices) != rank:
                raise RuntimeError("Number of indices does not match rank of coefficient. %d vs %d." % (len(dimIndices), rank))
            indices.extend(dimIndices)

        return indices

    def subscript_SpatialDerivative(self,tree):
        operand, _ = tree.operands()
        # Take the topmost element of the index stack
        dimIndices = self._indexStack.peek()
        indices = []
        # The first element is the dimension index corresponding to the derivative.
        # Push the remaining indices (which may be empty) on the stack
        self._indexStack.push(dimIndices[1:])
        # Append the indices of the operand (argument or coefficient)
        indices += self.subscript(operand)
        # The last element is the dimension index corresponding to the derivative.
        indices.append(dimIndices[0])
        # Restore the stack
        self._indexStack.pop()
        return indices

    def subscript_LocalTensor(self, form):
        form_data = form.form_data()
        rank = form_data.rank

        indices = []
        # One rank index for each rank
        for r in range(rank):
            indices.append(buildBasisIndex(r, form))

        return indices

    def buildCoeffQuadratureAccessor(self, coeff, fake_indices=False):
        rank = coeff.rank()
        if isinstance(coeff, Coefficient):
            name = buildCoefficientQuadName(coeff)
        else:
            # The spatial derivative adds an extra dim index so we need to
            # bump up the rank
            rank = rank + 1
            name = buildSpatialDerivativeName(coeff)
        base = Variable(name)

        # If there are no indices present (e.g. in the quadrature evaluation loop) then
        # we need to fake indices for the coefficient based on its rank:
        if fake_indices:
            fake = [buildDimIndex(i, coeff) for i in range(rank)]
            self._indexStack.push(tuple(fake))

        indices = self.subscript_Coefficient(coeff)

        # Remove the fake indices
        if fake_indices:
            self._indexStack.pop()

        coeffExpr = buildMultiArraySubscript(base, indices)
        return coeffExpr

    def buildLocalTensorAccessor(self, form):
        indices = self.subscript_LocalTensor(form)

        # Subscript the local tensor variable
        expr = self.buildSubscript(localTensor, indices)
        return expr

    def buildSubscript(self, variable, indices):
        raise NotImplementedError("You're supposed to implement buildSubscript()!")

    # We don't need to generate any code for a SubExpr node.
    # FIXME: It should be an error to encounter this, when the generation of
    # correctly-partitioned code is working.
    def sub_expr(self, se):
        return Variable("ST%s" % se.count())

    def component_tensor(self, tree, *ops):
        # We ignore the 2nd operand (a MultiIndex)
        return ops[0]

    # When entering an index, we need to memorise the indices that are attached
    # to it before descending into the sub-tree. The sub-tree handlers can make
    # use of these memorised indices in their code generation.
    def indexed(self, tree):
        o, i = tree.operands()
        self._indexStack.push(self.visit(i))
        op = self.visit(o)
        self._indexStack.pop()
        return op

    def multi_index(self, tree):
        indices = []
        for i in tree:
            if isinstance(i, Index):
                dims = tree.index_dimensions()
                indices.append(buildDimIndex(i.count(), dims[i]))
            elif isinstance(i, FixedIndex):
                indices.append(buildConstDimIndex(i._value))
            else:
                raise RuntimeError('Other types of indices not yet implemented.')
        return tuple(indices)


    # We need to keep track of how many IndexSums we passed through
    # so that we know which dim index we're dealing with.
    def index_sum(self, tree):
        o, i = tree.operands()
        self._sumIndexStack.push(self.visit(i))
        op = self.visit(o)
        self._sumIndexStack.pop()
        return op

    def list_tensor(self, tree):
        dimIndices = self._indexStack.peek()

        # Get expressions for all operands of the ListTensor in right order
        self._indexStack.push(())
        init = InitialiserList([self.visit(o) for o in tree.operands()])
        self._indexStack.pop()

        # If we have dimension indices on the stack we're right below an indexed
        if len(dimIndices) > 0:
            # Use IndexSum indices to build and subscript ListTensor, s.t.
            # indices match those of the loop nest
            subscriptIndices = [i[0] for i in self._sumIndexStack[-len(dimIndices):]]
            subscriptIndices.reverse()
            tmpTensor = buildListTensorVar(subscriptIndices)

            # Build the expressions populating the components of the list tensor
            decl = Declaration(tmpTensor)
            self._subExprStack.append(AssignmentOp(decl, init))

            # Build a subscript for the temporary Array and push that on the
            # expression stack
            return buildMultiArraySubscript(tmpTensor, subscriptIndices)
        # Otherwise we're operand of a higher rank ListTensor
        else:
            return init

    def constant_value(self, tree):
        if isinstance(tree, SymbolicValue):
            return Variable(tree.value())
        else:
            return Literal(tree.value())

    def sum(self, tree, *ops):
        return AddOp(*ops)

    def product(self, tree, *ops):
        return MultiplyOp(*ops)

    def division(self, tree, *ops):
        return DivideOp(*ops)

    def spatial_derivative(self, tree):
        name = buildSpatialDerivativeName(tree)
        indices = self.subscript(tree)
        return buildMultiArraySubscript(Variable(name), indices)

    def argument(self, tree):
        e = tree.element()
        indices = self.subscript(tree)

        if isinstance(e, FiniteElement):
            base = Variable(buildArgumentName(tree))
        elif isinstance(e, VectorElement):
            base = Variable(buildVectorArgumentName(tree))
        else:
            base = Variable(buildTensorArgumentName(tree))
        return buildMultiArraySubscript(base, indices)

    def coefficient(self, tree):
        return self.buildCoeffQuadratureAccessor(tree)

class QuadratureExpressionBuilder:

    def __init__(self, formBackend):
        self._formBackend = formBackend

    def build(self, tree):
        # Build Accessor for values at nodes
        coeffIndices = self.subscript(tree)
        
        if isinstance(tree, Coefficient):
            coeffName = buildCoefficientName(tree)
            argName = buildArgumentName(tree)
            argIndices = self.subscript_argument(tree)

            # Check if we are dealing with the Jacobian
            if isJacobian(tree):
                # Subscript coordinate field if we are dealing with the Jacobian
                coeffIndices = self.subscript(extractCoordinates(tree))
                # We actually need to pass a SpatialDerivative to build its name
                # FIXME: can this be done any nicer?
                from ufl.objects import i
                fakeArg = TrialFunction(extract_element(tree))
                fakeDerivative = SpatialDerivative(fakeArg,i)
                argName = buildSpatialDerivativeName(fakeDerivative)
                # Add an index over dimensions, since we're dealing with shape
                # derivatives (Important: build an index over the 2nd dimension)
                argIndices += [buildDimIndex(1,tree)]

        elif isinstance (tree, SpatialDerivative):
            operand, indices = tree.operands()
            coeffName = buildCoefficientName(operand)

            basis = TrialFunction(operand.element())
            basisDerivative = SpatialDerivative(basis, indices)
            argName = buildSpatialDerivativeName(basisDerivative)
            argIndices = self.subscript_argument_derivative(basisDerivative)

            # Check if we are dealing with the Jacobian
            if isinstance(operand.element().quadrature_scheme(), Coefficient):
                raise RuntimeError("Oops, the Jacobian shouldn't appear under a derivative.")

        coeffExpr = self.buildSubscript(Variable(coeffName), coeffIndices)
        argExpr = buildMultiArraySubscript(Variable(argName), argIndices)

        # Combine to form the expression
        expr = MultiplyOp(coeffExpr, argExpr)
        return expr

    def buildSubscript(self, variable, indices):
        raise NotImplementedError("You're supposed to implement buildSubscript()!")

    def subscript(self, tree):
        raise NotImplementedError("You're supposed to implement subscript()!")

    def subscript_argument(self, tree):
        # The count of the basis function induction variable is always
        # 0 in the quadrature loops (i.e. i_r_0)
        # We use the scalar basis for the expression evaluating a coefficient
        # at the quadrature points even if the coefficient is on a vector or
        # tensor basis since that is (in UFL) by definition a tensor product of
        # the scalar basis. So we need to extract the sub element.
        # FIXME: This will break for mixed elements
        indices = [buildBasisIndex(0, extract_subelement(tree)),
                   buildGaussIndex(self._formBackend.numGaussPoints)]
        return indices

    def subscript_argument_derivative(self, tree):
        # The count of the basis function induction variable is always
        # 0 in the quadrature loops (i.e. i_r_0), and only the first dim
        # index should be used to subscript the derivative (I think).
        return self.subscript_argument(tree.operands()[0]) \
                + [ buildDimIndex(0,tree) ]

def buildListTensorVar(indices):
    # FIXME: is that a stable naming scheme?
    name = 'l'+''.join([str(i._count) for i in indices])
    t = Array(Real(), [i.extent() for i in indices])
    return Variable(name, t)

# vim:sw=4:ts=4:sts=4:et
