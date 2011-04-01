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
import form
from codegeneration import Variable, Integer, Pointer, Real
from utilities import uniqify
# FEniCS UFL libs
from ufl.algorithms.transformations import Transformer

numElements = Variable("n_ele", Integer() )
statutoryParameters = [ form.localTensor, numElements, form.timestep, form.detwei ]

# For generating the kernel parameter list

class KernelParameterComputer(Transformer):

    def compute(self, tree):
        parameters = list(statutoryParameters)
        
        self._coefficients = []
        self._arguments = []
        self._spatialDerivatives = []
        
        self.visit(tree)
        
        self._coefficients = uniqify(self._coefficients)
        self._arguments = uniqify(self._arguments)
        self._spatialDerivatives = uniqify(self._spatialDerivatives)
        
        parameters.extend(self._coefficients)
        parameters.extend(self._arguments)
        parameters.extend(self._spatialDerivatives)
        return parameters

    # The expression structure does not affect the parameters.
    def expr(self, tree, *ops):
        pass

    def spatial_derivative(self, tree):
        name = form.buildSpatialDerivativeName(tree)
        parameter = Variable(name, Pointer(Real()))
        self._spatialDerivatives.append(parameter)

    def argument(self, tree):
        name = form.buildArgumentName(tree)
        parameter = Variable(name, Pointer(Real()))
        self._arguments.append(parameter)

    def coefficient(self, tree):
        name = form.buildCoefficientName(tree)
        parameter = Variable(name, Pointer(Real()))
        self._coefficients.append(parameter)


