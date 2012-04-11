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
from ufl.coefficient import Coefficient

ufl_elements = [FiniteElement, VectorElement, TensorElement]

class field_dict(dict):
    """Used for the state.xxx_fields dicts. It functions like a regular dict,
       but remembers which keys are set and get when in the 'run' mode."""

    def __init__(self, rank = 0):
        self._rank = rank
        self._run = False
        self._accessedFields = {}
        self._returnedFields = {}
        dict.__init__(self)

    def readyToRun(self):
        """Call this when the dict has been set up with the fields, before
           the ufl input runs"""
        self._run = True

    def __getitem__(self, field):
        coeff = dict.__getitem__(self, field)
        if self._run:
            self._accessedFields[coeff.count()] = (self._rank, field)
        return coeff

    def __setitem__(self, field, coeff):
        # Sanity check: only Coefficients can be written back to state
        assert isinstance(coeff, Coefficient), "Only Coefficients can be written back to state"
        if self._run:
            self._returnedFields[coeff.count()] = (self._rank, field)
        dict.__setitem__(self, field, coeff)

class UflState:

    def __init__(self):
        self.scalar_fields = field_dict(0)
        self.vector_fields = field_dict(1)
        self.tensor_fields = field_dict(2)

    def __getitem__(self,key):

        if isinstance(key,int):
            if key == 0:
                return self.scalar_fields
            elif key == 1:
                return self.vector_fields
            elif key == 2:
                return self.tensor_fields
            else:
                raise IndexErrror

        else:
            raise TypeErrror

    def __repr__(self):
        return str(self.scalar_fields)+', '+str(self.vector_fields)+', '+str(self.tensor_fields)

    def accessedFields(self):
        "A dict of all fields which have been retrieved from state"
        accessedFields = {}
        for rank in [0, 1, 2]:
            accessedFields.update(self[rank]._accessedFields)
        return accessedFields

    def returnedFields(self):
        "A dict of all fields which have been written back to state"
        returnedFields = {}
        for rank in [0, 1, 2]:
            returnedFields.update(self[rank]._returnedFields)
        return returnedFields

    # FIXME hard coded for triangles
    def insert_field(self, field, rank, shape = 'CG', degree = 1, quad_degree = 4):
        "Insert field as a coefficient of the element type given by rank, shape and degree"
        self[rank][field] = Coefficient(ufl_elements[rank](shape, cell="triangle", degree=degree, quad_scheme=quad_degree))

    def readyToRun(self):
        self.scalar_fields.readyToRun()
        self.vector_fields.readyToRun()
        self.tensor_fields.readyToRun()

# vim:sw=4:ts=4:sts=4:et
