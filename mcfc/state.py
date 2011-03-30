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


import ufl.finiteelement

# Provides access to the syntax for getting variables from state

class TemporalIndex():

    def __init__(self, offset=0):
        self._offset = offset

    def __add__(self, other):
        return TemporalIndex(other)

    def __radd__(self, other):
        return TemporalIndex(other)

    def __sub__(self, other):
        return TemporalIndex(other*(-1))

    def __rsub__(self, other):
        return TemporalIndex(other*(-1))

    def getOffset(self):
        return self._offset

    def isConstant(self):
        return False

class ConstantTemporalIndex(TemporalIndex):
    
    def isConstant(self):
        return True

    def __add__(self, other):
        raise RuntimeError("The constant temporal index cannot be offset")

    def __radd__(self, other):
        raise RuntimeError("The constant temporal index cannot be offset")

    def __sub__(self, other):
        raise RuntimeError("The constant temporal index cannot be offset")

    def __rsub__(self, other):
        raise RuntimeError("The constant temporal index cannot be offset")

    def getOffset(self):
        return 0

class __fake_field_dict__():
    def __init__(self,rank):
        self.rank=rank
        
    def __getitem__(self,key):
        
        field_name, timestep = key

        if (self.rank==0):
            degree = _finiteElements[field_name]
            field = ufl.finiteelement.FiniteElement("Lagrange", "triangle", degree)
        elif(self.rank==1):
            degree = _vectorElements[field_name]
            field = ufl.finiteelement.VectorElement("Lagrange", "triangle", degree)
        elif(self.rank==2):
            degree = _tensorElements[field_name]
            field = ufl.finiteelement.TensorElement("Lagrange", "triangle", degree)

        return field

    def __setitem__(self,key,data):

        field_name, timestep = key


scalar_fields = __fake_field_dict__(0)
vector_fields = __fake_field_dict__(1)
tensor_fields = __fake_field_dict__(2)

# We need to know the basis for each field in code generation. This should be
# read from an flml file - however, for now for testing, just a list of the
# usual names I use and their order will do.

_finiteElements = { 'Tracer': 1 }
_vectorElements = { 'Velocity': 1, 'NewVelocity': 1 }
_tensorElements = { 'TracerDiffusivity': 1 }

def getRank(field):
    if field in _finiteElements.keys():
        return 0
    elif field in _vectorElements.keys():
        return 1
    elif field in _tensorElements.keys():
        return 2
    else:
        return -1
        
# vim:sw=4:ts=4:sts=4:et
