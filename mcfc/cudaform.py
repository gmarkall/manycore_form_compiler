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
from cudaparameters import CudaKernelParameterGenerator, numElements, statutoryParameters
from cudaexpression import CudaExpressionBuilder, CudaQuadratureExpressionBuilder, buildElementLoop, ElementIndex
# UFL libs
from ufl.finiteelement import FiniteElement, VectorElement, TensorElement

class CudaFormBackend(FormBackend):

    def __init__(self):
        FormBackend.__init__(self)
        self._expressionBuilder = CudaExpressionBuilder(self)
        self._quadratureExpressionBuilder = CudaQuadratureExpressionBuilder(self)

    def compile(self, name, form):
        "Compile a form with a given name."

        # The element loop is the outermost loop
        outerLoop = buildElementLoop()

        # Build the kernel as normal, but pass the element loop as the outer
        # scope to nest the other loop nests under
        kernel = super(CudaFormBackend, self).compile(name, form, outerLoop)

        # Make this a Cuda kernel.
        kernel.setCudaKernel(True)
        return kernel

    def _buildCoeffQuadDeclaration(self, name, rank):
        length = self.numGaussPoints * pow(self.numDimensions, rank)
        return Declaration(Variable(name, Array(Real(), Literal(length))))

    def _buildKernelParameters(self, tree, form):
        KPG = CudaKernelParameterGenerator()
        return KPG.generate(tree, form, statutoryParameters)

# vim:sw=4:ts=4:sts=4:et
