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

from expression import *

def buildSubscript(variable, indices):
    """Given a list of indices, return an AST that subscripts
    the given array using those indices. The order is
    important."""

    # Add subscripts for all indices
    for i in indices:
        variable = Subscript(variable, Variable(i.name()))

    return variable

# Expression builders

class Op2ExpressionBuilder(ExpressionBuilder):

    def buildSubscript(self, variable, indices):
        return buildSubscript(variable, indices)

    def subscript_LocalTensor(self, form_data):
        # For matrices the local tensor variable is a pointer to a single entry.
        if form_data.rank == 2:
            i = []
        else:
            i = super(Op2ExpressionBuilder,self).subscript_LocalTensor(form_data)

        i.append(buildConstDimIndex(0))
        return i

    def symbolic_value(self, value):
        return Bracketed(Dereference(Variable(value)))

class Op2QuadratureExpressionBuilder(QuadratureExpressionBuilder):

    def buildSubscript(self, variable, indices):
        return buildSubscript(variable, indices)

    def subscript(self, tree):
        # The OP2 specification states that a vector-valued coefficient be
        # indexed by separate indices for the scalar basis and the spatial
        # dimension(s) (same for tensor-valued coefficients). Hence we need to
        # extract the scalar element to build the appropriate basis index.
        indices = [buildQuadratureBasisIndex(0, extract_subelement(tree))]
        
        # Scalars are actually an array of length 1
        if tree.rank() == 0:
            indices.append(buildConstDimIndex(0))
        
        # This deals with the general case (vectors, tensors)
        for r in range(tree.rank()):
            indices.append(buildDimIndex(r,tree))
        return indices
    
# vim:sw=4:ts=4:sts=4:et
