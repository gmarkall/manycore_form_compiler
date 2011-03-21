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
