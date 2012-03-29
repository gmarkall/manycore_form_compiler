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


# MCFC libs
from form import *
from op2expression import Op2ExpressionBuilder, Op2QuadratureExpressionBuilder

class Op2FormBackend(FormBackend):

    def __init__(self):
        FormBackend.__init__(self)
        self._expressionBuilder = Op2ExpressionBuilder()
        self._quadratureExpressionBuilder = Op2QuadratureExpressionBuilder()

    def _buildCoefficientParameter(self, coeff):
        # Use the coordinate field instead of the Jacobian when building the
        # subscript.
        indices = self._quadratureExpressionBuilder.subscript(extractCoordinates(coeff))
        # Do however use the Jacobian coefficent when building the name, since
        # the coordinate coefficient doesn't get renumbered!
        name = buildCoefficientName(coeff)
        return buildPtrArrParameter(name, indices)

    def _buildLocalTensorParameter(self, form):
        n = localTensor.name()
        # OP2 needs a * for a local matrix or a ** for a local vector.
        # (Yes, really).
        if form.form_data().rank == 2:
            return buildPtrParameter(n)
        else: 
            return buildPtrPtrParameter(n)

    def buildLocalTensorInitialiser(self, form):
        # For matrices the local tensor doesn't need to be initialised.
        if form.form_data().rank == 2:
            return NullExpression();
        else:
            return super(Op2FormBackend, self).buildLocalTensorInitialiser(form)

    def _buildKernelParameters(self, form):
        p = super(Op2FormBackend, self)._buildKernelParameters(form)
        # For a local matrix, we need parameters for the op2 iteration space
        if form.form_data().rank == 2:
            p.extend([Variable("i", Integer()), Variable("j", Integer())])
        return p

def buildPtrParameter(name):
    return Variable(name, Pointer(Real()))

def buildPtrPtrParameter(name):
    return Variable(name, Pointer(Pointer(Real())))

def buildPtrArrParameter(name, indices):
    return Variable(name, Pointer(Array(Real(), [i.extent() for i in indices[1:]])))

# vim:sw=4:ts=4:sts=4:et
