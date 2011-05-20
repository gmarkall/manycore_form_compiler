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

from ufl.finiteelement import FiniteElement, VectorElement, TensorElement

ufl_elements = [FiniteElement, VectorElement, TensorElement]

class UflState:

    def __init__(self):
        self.scalarfields = {}
        self.vectorfields = {}
        self.tensorfields = {}

    def __getitem__(self,key):

        if isinstance(key,int):
            if key == 0:
                return self.scalarfields
            elif key == 1:
                return self.vectorfields
            elif key == 2:
                return self.tensorfields
            else:
                raise IndexErrror

        else:
            raise TypeErrror

    def __repr__(self):
        return str(self.scalarfields)+', '+str(self.vectorfields)+', '+str(self.tensorfields)

    def insert_field(self, field, rank, shape = 'CG', degree = 1):
        self[rank][field] = ufl_elements[rank](shape, "triangle", degree)

# vim:sw=4:ts=4:sts=4:et
