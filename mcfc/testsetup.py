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


from ufl import *
from ufl.algebra import *
from ufl.differentiation import *
from ufl.indexed import *
from ufl.indexsum import *
from ufl.tensors import *
from ufl.algorithms.preprocess import *
HH = Form([Integral(Sum(IndexSum(Product(Indexed(ComponentTensor(SpatialDerivative(Argument(FiniteElement('Lagrange', Cell('triangle', 1, Space(2)), 1), 0), MultiIndex((Index(0),), {Index(0): 2})), MultiIndex((Index(0),), {Index(0): 2})), MultiIndex((Index(1),), {Index(1): 2})), Indexed(ComponentTensor(SpatialDerivative(Argument(FiniteElement('Lagrange', Cell('triangle', 1, Space(2)), 1), 1), MultiIndex((Index(2),), {Index(2): 2})), MultiIndex((Index(2),), {Index(2): 2})), MultiIndex((Index(1),), {Index(1): 2}))), MultiIndex((Index(1),), {Index(1): 2})), Product(IntValue(-1, (), (), {}), Product(Argument(FiniteElement('Lagrange', Cell('triangle', 1, Space(2)), 1), 0), Argument(FiniteElement('Lagrange', Cell('triangle', 1, Space(2)), 1), 1)))), Measure('cell', 0, None))])
RHS = Form([Integral(Product(Argument(FiniteElement('Lagrange', Cell('triangle', 1, Space(2)), 1), 0), Coefficient(FiniteElement('Lagrange', Cell('triangle', 1, Space(2)), 1), 0)), Measure('cell', 0, None))])

A = Form([Integral(Sum(Product(Argument(FiniteElement('Lagrange', Cell('triangle', 1, Space(2)), 1), 0), Argument(FiniteElement('Lagrange', Cell('triangle', 1, Space(2)), 1), 1)), Product(IntValue(-1, (), (), {}), Product(FloatValue(0.5, (), (), {}), IndexSum(Product(IndexSum(Product(Indexed(Coefficient(TensorElement('Lagrange', Cell('triangle', 1, Space(2)), 1, (2, 2), None), 0), MultiIndex((Index(0), Index(1)), {Index(0): 2, Index(1): 2})), Product(IntValue(-1, (), (), {}), Indexed(ComponentTensor(SpatialDerivative(Argument(FiniteElement('Lagrange', Cell('triangle', 1, Space(2)), 1), 0), MultiIndex((Index(2),), {Index(2): 2})), MultiIndex((Index(2),), {Index(2): 2})), MultiIndex((Index(0),), {Index(0): 2})))), MultiIndex((Index(0),), {Index(0): 2})), Indexed(ComponentTensor(SpatialDerivative(Argument(FiniteElement('Lagrange', Cell('triangle', 1, Space(2)), 1), 1), MultiIndex((Index(3),), {Index(3): 2})), MultiIndex((Index(3),), {Index(3): 2})), MultiIndex((Index(1),), {Index(1): 2}))), MultiIndex((Index(1),), {Index(1): 2}))))), Measure('cell', 0, None))])

import cudaform
import cudaassembler
import frontend

source = frontend.readSource('diffusion-1.cufl')

CFB = cudaform.CudaFormBackend()

# vim:sw=4:ts=4:sts=4:et
