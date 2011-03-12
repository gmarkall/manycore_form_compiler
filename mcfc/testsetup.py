from ufl import *
from ufl.algebra import *
from ufl.differentiation import *
from ufl.indexed import *
from ufl.indexsum import *
from ufl.tensors import *
from ufl.algorithms.preprocess import *
HH = Form([Integral(Sum(IndexSum(Product(Indexed(ComponentTensor(SpatialDerivative(Argument(FiniteElement('Lagrange', Cell('triangle', 1, Space(2)), 1), 0), MultiIndex((Index(0),), {Index(0): 2})), MultiIndex((Index(0),), {Index(0): 2})), MultiIndex((Index(1),), {Index(1): 2})), Indexed(ComponentTensor(SpatialDerivative(Argument(FiniteElement('Lagrange', Cell('triangle', 1, Space(2)), 1), 1), MultiIndex((Index(2),), {Index(2): 2})), MultiIndex((Index(2),), {Index(2): 2})), MultiIndex((Index(1),), {Index(1): 2}))), MultiIndex((Index(1),), {Index(1): 2})), Product(IntValue(-1, (), (), {}), Product(Argument(FiniteElement('Lagrange', Cell('triangle', 1, Space(2)), 1), 0), Argument(FiniteElement('Lagrange', Cell('triangle', 1, Space(2)), 1), 1)))), Measure('cell', 0, None))])
HH = preprocess(HH)
RHS = Form([Integral(Product(Argument(FiniteElement('Lagrange', Cell('triangle', 1, Space(2)), 1), 0), Coefficient(FiniteElement('Lagrange', Cell('triangle', 1, Space(2)), 1), 0)), Measure('cell', 0, None))])
RHS = preprocess(RHS)

import cuda
