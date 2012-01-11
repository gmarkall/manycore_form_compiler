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
from op2parameters import Op2KernelParameterGenerator, _buildArrayParameter
from op2expression import Op2ExpressionBuilder, Op2QuadratureExpressionBuilder

class Op2FormBackend(FormBackend):

    def __init__(self):
        FormBackend.__init__(self)
        self._expressionBuilder = Op2ExpressionBuilder(self)
        self._quadratureExpressionBuilder = Op2QuadratureExpressionBuilder(self)

    def _buildKernelParameters(self, tree, form):
        KPG = Op2KernelParameterGenerator(self)

        timestep = Variable("dt", Real() )
        localTensor = _buildArrayParameter("localTensor", self._expressionBuilder.subscript_LocalTensor(form))

        statutoryParameters = [ localTensor, timestep ]

        return KPG.generate(tree, form, statutoryParameters)

# vim:sw=4:ts=4:sts=4:et
